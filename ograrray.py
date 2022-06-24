import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from random import randint
from datetime import datetime

ora = datetime.now().strftime('%Y%m%d%H%M%S')
local = False if 'SSH_CONNECTION' in os.environ else True
compute_name = os.uname().nodename
interpreters = {
    'ammarella': '/home/aba/anaconda3/envs/canomaly/bin/python',
    'dragon': '',
    'goku': '/usr/local/anaconda3/envs/canomaly/bin/python',
    'rb_torch': '/opt/conda/envs/canomaly/bin/python'
}
paths = {
    'ammarella': '',
    'dragon': '',
    'goku': '/home/aba/deploy-rb',
    'rb_torch': '/home/energy/deploy-rb'
}

# launch this inside canomaly

parser = ArgumentParser(description='ograrray', allow_abbrev=False)
parser.add_argument('--path', type=str, required=True, help='list_path')
parser.add_argument('--script-to-run', type=str, required=True, help='path to script to run', default='utils/main.py')
parser.add_argument('--nickname', type=str, default=None, help='Job name.')
parser.add_argument('--cycles', type=int, default=1,
                    help='The number of cycles.')
parser.add_argument('--max_jobs', type=int, default=None,
                    help='The maximum number of jobs.')
parser.add_argument('--skip_first', type=int, default=0,
                    help='Skips the first skip_first jobs.')
parser.add_argument('--exclude', type=str, default='', help='excNodeList.')
parser.add_argument('--n_gpus', type=int, default=1, help='Number of requested GPUs per job')
parser.add_argument('--n_nodes', type=int, default=1, help='Number of nodes')
parser.add_argument('--n_proc_per_node', type=int, default=1, help='Number of process per node')
parser.add_argument('--custom_time', default=None, type=str, help='customizes sbatch time')
# parser.add_argument('--jobs_per_gpu', default=1, type=int, help='customizes sbatch time')
parser.add_argument('--user_name', type=str, default='rbenaglia')
parser.add_argument('--envname', type=str, default='env1')
parser.add_argument('--sbacciu', action='store_true')
parser.add_argument('--parallel_process', action='store_true')
parser.add_argument('--multiprocess', action="store_true")
# parser.add_argument('--logs', action='store_true')
# parser.add_argument('--rand_ord', action='store_true')
args = parser.parse_args()

red = int(args.cycles)

with open(args.path) as f:
    sbacci = f.read().splitlines()

if not local:
    if compute_name in ('ammarella', 'dragon', 'goku', 'rb_torch'):
        interpreter = interpreters.get(compute_name, None)
    else:
        interpreter = '/homes/%s/.conda/envs/%s/bin/python' % (args.user_name, args.envname)
else:
    interpreter = sys.executable

if args.multiprocess:
    interpreter += f' -m torch.distributed.launch --nproc_per_node {args.n_proc_per_node}'
assert interpreter is not None

sbacci = [f'\'{interpreter} {args.script_to_run} ' + x + '\'' for x in sbacci if
          not x.startswith('#') and len(x.strip())] * red
sbacci = sbacci[args.skip_first:]

max_jobs = args.max_jobs if args.max_jobs is not None else len(sbacci)
nickname = args.nickname if args.nickname is not None else 'der-verse'

if not local and (compute_name not in ('ammarella', 'dragon', 'goku', 'rb_torch')):
    # if args.jobs_per_gpu == 1:
    gridsh = '''#!/bin/bash
#SBATCH -p prod
#SBATCH --job-name=<nick>
#SBATCH --array=0-<lung>%<mj>
#SBATCH --nodes=<nnodes>
<time>
#SBATCH --output="/homes/<user_name>/output/<nick>_%A_%a.out"
#SBATCH --error="/homes/<user_name>/output/<nick>_%A_%a.err"
#SBATCH --gres=gpu:<ngpu>
<xcld>

arguments=(
<REPLACEME>
)

sleep $(($RANDOM % 20)); ${arguments[$SLURM_ARRAY_TASK_ID]}
'''
    gridsh = gridsh.replace('<REPLACEME>', '\n'.join(sbacci))
    gridsh = gridsh.replace('<lung>', str(len(sbacci) - 1))
    gridsh = gridsh.replace('<xcld>', ('#SBATCH --exclude=' + args.exclude) if len(args.exclude) else '')
    gridsh = gridsh.replace('<nick>', nickname)
    gridsh = gridsh.replace('<mj>', str(max_jobs))
    gridsh = gridsh.replace('<ngpu>', str(args.n_gpus))
    gridsh = gridsh.replace('<nnodes>', str(args.n_nodes))
    gridsh = gridsh.replace('<user_name>', args.user_name)
    gridsh = gridsh.replace('<time>', ('#SBATCH --time=' + args.custom_time) if args.custom_time is not None else '')

    with open('/homes/rbenaglia/sbatch/ready_for_sbatch.sh', 'w') as f:
        f.write(gridsh)
    print(gridsh)
    if args.sbacciu:
        os.system('sbatch /homes/rbenaglia/sbatch/ready_for_sbatch.sh')

else:
    if args.sbacciu:
        if args.parallel_process and not local:
            base_path = os.path.join(paths.get(compute_name, '.'), 'outputs')
            if not os.path.isdir(base_path):
                os.mkdir(base_path)
            for num, c in enumerate(sbacci):
                # with open(os.path.join(base_path, f"{ora}_{num}.out"), "wb") as out, open(
                #         os.path.join(base_path, f"{ora}_{num}.err"),
                #         "wb") as err:
                # subprocess.Popen("ls", stdout=out, stderr=err)

                # cmd = f'nohup bash -c "exec -a process_{ora}_{num} ' + c.replace("'", '') + '" > ' + \
                #       os.path.join(base_path, f"{ora}_{num}.out") + ' &'
                cmd = 'nohup ' + c.replace("'", '') + ' > ' + os.path.join(base_path, f"{ora}_{num}.out") + ' &'
                print(cmd)
                res = subprocess.Popen(
                    cmd,
                    shell=True, cwd=os.path.join(paths.get(compute_name, '.'), 'mammoth')
                    # check=True
                )
                time.sleep(5)
        else:
            for c in sbacci:
                res = subprocess.run(
                    c.replace("'", ''), shell=True,
                    check=True
                )
