from arguments import get_args
import torch
from utils.dist import setup_for_distributed
from engine.trainer import run_training
from engine.worker import spawn_workers
from engine.trainer import run_linear_eval_and_finalize

if __name__ == "__main__":
    args = get_args()

    # default dist backend
    if not hasattr(args, "dist_backend"):
        args.dist_backend = "nccl"

    # No multi-processing → single-process (DP or single GPU)
    if not getattr(args, "multiprocessing_distributed", False):
        args.distributed = False
        args.device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        setup_for_distributed(True)

        print("Arguments:")
        for k, v in vars(args).items():
            print(f"{k}: {v}")

        run_training(args)
        run_linear_eval_and_finalize(args)
    else:
        # mp.spawn on all visible GPUs
        print("Arguments:")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        spawn_workers(args)