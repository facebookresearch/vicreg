# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import submitit
import os
import uuid
from pathlib import Path

import main_vicreg


def parse_args():
    parser = argparse.ArgumentParser(
        "Submitit for VICReg", parents=[main_vicreg.get_arguments()]
    )
    parser.add_argument(
        "--nodes", default=4, type=int, help="Number of nodes to request"
    )
    parser.add_argument(
        "--ngpus", default=8, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--timeout", default=72, type=int, help="Duration of the job, in hours"
    )
    parser.add_argument("--job_name", default="vicreg", type=str, help="Job name")
    parser.add_argument(
        "--partition", default="mypartition", type=str, help="Partition where to submit"
    )
    parser.add_argument(
        "--constraint",
        default="",
        type=str,
        help="Slurm constraint. Use 'volta32gb' for Tesla V100 with 32GB",
    )
    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/vicreg")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def main():
    args = parse_args()

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.exp_dir)

    executor.update_parameters(
        name=args.job_name,
        mem_gb=40 * args.ngpus,
        gpus_per_node=args.ngpus,
        tasks_per_node=args.ngpus,
        cpus_per_task=10,
        nodes=args.nodes,
        timeout_min=args.timeout * 60,
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        slurm_constraint=args.constraint,
        slurm_comment=args.comment,
    )

    args.dist_url = get_init_file().as_uri()
    print("Distributed URL: ", args.dist_url)

    job = executor.submit(main_vicreg.main, args)
    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
