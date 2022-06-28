# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import inspect
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets
from tqdm import tqdm

import augmentations as aug
from distributed import init_distributed_mode

import resnet
import hubconf as hc
from logger import logger
from spectral_analysis import laplacian_analysis
from folder import ImageFolder


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
                        help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--avoid-dist', action='store_true')

    # Spectral analysis
    parser.add_argument('--restart_from_checkpoint', action='store_true')
    parser.add_argument('--distance_metric', type=str, default='euclid', choices=['euclid', 'cosine', 'weiss'])
    parser.add_argument('--knn', type=int, default=0)
    parser.add_argument('--sigma', type=float, default=1.)
    parser.add_argument('--debug_spectral', action='store_true')
    parser.add_argument('--finetune_spectral', action='store_true')
    return parser


def isdebugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


def main(args):
    assert not args.finetune_spectral or not args.debug_spectral, "Cannot use both --debug_spectral and --finetune_spectral"

    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    logger.log(args)
    gpu = torch.device(args.device)
    complete_logger = {}
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    complete_logger["args"] = vars(args)
    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        logger.log(" ".join(sys.argv))

    transforms = aug.TrainTransform()

    training_dataset = "train"  # if not isdebugging() else "imagenet_fake"

    dataset = ImageFolder(args.data_dir / training_dataset, transforms)
    logger.log("Load the dataset")
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    logger.log("Create sampler")

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=1,
        shuffle=True if sampler is None else False,
    )
    logger.log("Create loader")
    model = VICReg(args).cuda(gpu)
    if args.restart_from_checkpoint:
        model.backbone = hc.__dict__[args.arch.replace('x', 'w')](pretrained=True).to(gpu)
    if dist.is_initialized():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    logger.log("Create model")
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )
    logger.log("Create optimizer")
    if (args.exp_dir / "model.pth").is_file() and not args.restart_from_checkpoint:
        if args.rank == 0:
            logger.log("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        
        for k in list(ckpt["model"].keys()):
            if k.startswith('module.'):
                ckpt["model"][k.replace('module.', '')] = ckpt["model"][k]
                del ckpt["model"][k]

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()

    if args.debug_spectral:

        logger.log("debugging spectral analysis")

        with torch.no_grad():
            model.eval()
            complete_logger[f"results"] = {}
            l = next(iter(loader))
            x, y = l[0][0], l[0][1]
            xs, ys = [], []
            for virtual_batch in range(4):
                ox = x[virtual_batch*len(x) // 4:(virtual_batch+1)*len(x) // 4]
                oy = y[virtual_batch*len(x) // 4:(virtual_batch+1)*len(x) // 4]

                ox = model.backbone(ox.cuda(gpu, non_blocking=True)) # model.projector(y)
                xs.append(ox.cpu())
                logger.log(f'vb {virtual_batch} - first inference done')
                oy = model.backbone(oy.cuda(gpu, non_blocking=True))
                ys.append(oy.cpu())
                logger.log(f'vb {virtual_batch} - Inference done')
            
            # enter the cpu_zone
            xs, ys = torch.cat(xs), torch.cat(ys)
            
            for k in tqdm(range(2, args.knn, 2)):
                logger.log(f"knn {k}")        
                energy_1, eigenvalues_1, eigenvectors_1, L_1, (A_1, D_1, distances_1) = laplacian_analysis(
                    data=xs, sigma=1., knn=k, logvars=None, norm_lap=True, norm_eigs=False, n_pairs=0,
                    distance_metric=args.distance_metric)
                energy_2, eigenvalues_2, eigenvectors_2, L_2, (A_2, D_2, distances_2) = laplacian_analysis(
                    data=ys, sigma=1., knn=k, logvars=None, norm_lap=True, norm_eigs=False, n_pairs=0,
                    distance_metric=args.distance_metric)
                logger.log('eigenvectors calculation done')
                print(f"{k}-NN: fun_map-loss 1.T@2: "
                                            f"{fun_map_loss(eigenvectors_1, eigenvectors_2):.4f} "
                                            f"2.T@1:{fun_map_loss(eigenvectors_2, eigenvectors_1):.4f}")
                complete_logger[f"results"][f"{k}-NN"] = {'e1': eigenvalues_1.tolist(),
                                                            'e2': eigenvalues_2.tolist()}
            
            assert False # don't overwrite old results please
            logger.log(complete_logger, file=stats_file)
    elif args.finetune_spectral:
        logger.log('Start finetuning')
        for epoch in range(start_epoch, args.epochs):
            if dist.is_initialized():
                sampler.set_epoch(epoch)
            progress = logger.get_tqdm(enumerate(loader, start=epoch * len(loader)),
                                       f'TRAIN - epoch {epoch + 1}/{args.epochs}',
                                       leave=True)
            for step, ((x, y), _) in progress:
                x = x.cuda(gpu, non_blocking=True)
                y = y.cuda(gpu, non_blocking=True)

                lr = adjust_learning_rate(args, optimizer, loader, step)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = model.forward(x, y)
                progress.set_postfix(loss)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                exit()

                current_time = time.time()

                if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        loss=loss.item(),
                        time=int(current_time - start_time),
                        lr=lr,
                    )
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
                    last_logging = current_time
            if args.rank == 0:
                state = dict(
                    epoch=epoch + 1,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                )
                torch.save(state, args.exp_dir / "model.pth")
        if args.rank == 0:
            torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")
    else:
        logger.log('Start training')
        for epoch in range(start_epoch, args.epochs):
            if dist.is_initialized():
                sampler.set_epoch(epoch)
            progress = logger.get_tqdm(enumerate(loader, start=epoch * len(loader)),
                                       f'TRAIN - epoch {epoch + 1}/{args.epochs}',
                                       leave=True)
            for step, ((x, y), _) in progress:
                x = x.cuda(gpu, non_blocking=True)
                y = y.cuda(gpu, non_blocking=True)

                lr = adjust_learning_rate(args, optimizer, loader, step)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = model.forward(x, y)
                progress.set_postfix(loss)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                current_time = time.time()

                if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        loss=loss.item(),
                        time=int(current_time - start_time),
                        lr=lr,
                    )
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
                    last_logging = current_time
            if args.rank == 0:
                state = dict(
                    epoch=epoch + 1,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                )
                torch.save(state, args.exp_dir / "model.pth")
        if args.rank == 0:
            torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        print(dist.get_rank(), 'x', x.shape) # TODO piallami

        repr_loss = F.mse_loss(x, y)
        if dist.is_initialized():
            x = torch.cat(FullGatherLayer.apply(x), dim=0)
            y = torch.cat(FullGatherLayer.apply(y), dim=0)

        print(dist.get_rank(), 'x after gather', x.shape) # TODO piallami

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)

        print(dist.get_rank(), 'std_x', std_x.shape) # TODO piallami

        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        print(dist.get_rank(), 'cov_x', cov_x.shape) # TODO piallami
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        rodo_loss = self.calc_rodo_loss(x, y)
        loss = (
                self.args.sim_coeff * repr_loss
                + self.args.std_coeff * std_loss
                + self.args.cov_coeff * cov_loss
        )
        return loss

    def calc_rodo_loss(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculate the RODO loss between two tensors.
        """
        energy_1, eigenvalues_1, eigenvectors_1, L_1, (A_1, D_1, distances_1) = laplacian_analysis(
            data=x, sigma=1., knn=self.args.knn, logvars=None, norm_lap=True, norm_eigs=False, n_pairs=0,
            distance_metric=self.args.distance_metric)
        energy_2, eigenvalues_2, eigenvectors_2, L_2, (A_2, D_2, distances_2) = laplacian_analysis(
            data=y, sigma=1., knn=self.args.knn, logvars=None, norm_lap=True, norm_eigs=False, n_pairs=0,
            distance_metric=self.args.distance_metric)
        loss = fun_map_loss(eigenvectors_1, eigenvectors_2)
        return loss


def fun_map_loss(eig_1: torch.Tensor, eig_2: torch.Tensor):
    fun_map = (eig_1.T @ eig_2).abs()
    targets = torch.eye(fun_map.shape[0]).to(fun_map.device)
    return torch.square(fun_map - targets).sum()


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
            self,
            params,
            lr,
            weight_decay=0,
            momentum=0.9,
            eta=0.001,
            weight_decay_filter=None,
            lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
