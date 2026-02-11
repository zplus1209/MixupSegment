import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from utils.dist import setup_for_distributed, is_main_process
from utils.misc import set_seed, compute_local_batch
from engine.trainer import run_training
from engine.trainer import run_linear_eval_and_finalize


def _main_worker(gpu, ngpus_per_node, args):
    args.local_rank = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    args.distributed = True

    torch.cuda.setDevice(args.local_rank) if hasattr(torch.cuda, "setDevice") else torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if getattr(args, "seed", None) is None:
        args.seed = 42
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size * ngpus_per_node,
        rank=args.rank,
    )

    ngpus_per_node = torch.cuda.device_count()

    # Adaptive local batch
    local_batch = compute_local_batch(args, ngpus_per_node)

    # Broadcast sum of local batches to compute global_batch_per_step
    local_bs_t = torch.tensor([local_batch], dtype=torch.int64, device=args.device)
    dist.all_reduce(local_bs_t, op=dist.ReduceOp.SUM)
    global_batch_per_step = int(local_bs_t.item())
    world_sz = dist.get_world_size(); mean_bs = global_batch_per_step / float(world_sz)

    if is_main_process():
        print(f"[AdaptiveBatch] rank={args.rank} local_batch={local_batch} | global_batch_per_step={global_batch_per_step}")

    args._local_train_batch = local_batch
    args._global_batch_per_step = global_batch_per_step
    args._mean_bs = mean_bs

    if args.resume_eval_from is not None:
        args.eval_from = args.resume_eval_from
        args.resume_eval_from = None
    else:
        run_training(args)

    dist.barrier(); dist.destroy_process_group()

    # rank-0: linear eval + finalize logs
    if args.rank == 0:
        args.distributed = False
        args.multiprocessing_distributed = False
        if torch.cuda.is_available():
            torch.cuda.init()
        run_linear_eval_and_finalize(args)


def spawn_workers(args):
    ngpus_per_node = torch.cuda.device_count()
    setup_for_distributed(is_master=True)
    mp.spawn(_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
