#!/bin/bash
#PBS -A MultiActiveAI
#PBS -k doe
#PBS -l walltime=02:00:00
#PBS -l filesystems=home
#PBS -q full-node
#PBS -l select=1:system=thetagpu
#PBS -N onenode_vicreg
#PBS -m abe
#PBS -M yufeng.luo@anl.gov 
#PBS -k eod
#PBS -o /home/brookluo/training_output/vicreg.out
#PBS -e /home/brookluo/training_output/vicreg.err


data_path="/lus/theta-fs0/projects/MultiActiveAI/sage-cloud-data"
src_dir="/home/brookluo/anl-su23/vicreg-sage"
# this must be *.py
model_pyfile="vit_single_vicreg.py"
arch="vit_tiny"
container_file="nvcr_py3_2107.sif"
container_path="$src_dir/container"
chkpt_path="$src_dir/test/checkpoint_vit"
num_epoch=10
batch_size=16
base_lr=0.3

cmd_exec_dist="python -m torch.distributed.launch --nproc_per_node=8"
cmd_exec_single="python"

cmd_exec=$cmd_exec_single

# make directory if one not exist
[ ! -d $chkpt_path ] && mkdir $chkpt_path

cd $src_dir
module load conda/pytorch; conda activate
$cmd_exec $model_pyfile --data-dir $data_path \
    --exp-dir $chkpt_path --arch $arch --epochs $num_epoch \
    --batch-size $batch_size --base-lr $base_lr \
    # --rgb-image-size 600 800 --ir-image-size 252 336
