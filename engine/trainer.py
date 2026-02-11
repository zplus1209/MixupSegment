import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm

from tools import Logger, knn_monitor

from utils.dist import is_main_process
from utils.misc import SmoothedValue, set_seed, format_mem
from engine.data import build_datasets_and_loaders
from engine.modeling import build_model_optimizer_scheduler
from engine.checkpoint import resume_if_needed, save_epoch_checkpoint, save_best_checkpoint, save_final
from engine.prof import maybe_create_profiler
from engine.linear_eval import main as linear_eval
from augmentations.gpu_view import GPUViewGen


def _weight_const_for_rank(args):
    import torch.distributed as dist
    if not (hasattr(args, 'distributed') and args.distributed and dist.is_initialized()):
        return 1.0
    mean_bs = getattr(args, "_mean_bs", None)
    local_bs_const = getattr(args, "_local_train_batch", None) or args.train.batch_size
    if mean_bs is None:
        world_sz = dist.get_world_size()
        gb = getattr(args, "_global_batch_per_step", local_bs_const * world_sz)
        mean_bs = gb / float(world_sz)
    return float(local_bs_const) / max(1.0, float(mean_bs))


def run_training(args):
    # defaults
    if not hasattr(args, "debug_profile"):
        args.debug_profile = 0
    if not hasattr(args, "limit_train_steps"):
        args.limit_train_steps = 0
    if getattr(args, "seed", None) is None:
        args.seed = 42

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = getattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Build dataloaders
    train_loader, memory_loader, test_loader, steps_per_epoch = build_datasets_and_loaders(args, device)
    if is_main_process():
        print(f"[StepsPerEpoch] steps_per_epoch={steps_per_epoch} (min over ranks)")

    # Build model/optim/scheduler
    model, backbone_for_eval, optimizer, lr_scheduler, (base_lr, warmup_lr, final_lr) = \
        build_model_optimizer_scheduler(args, steps_per_epoch, device)

    if is_main_process():
        gpus_per_node = torch.cuda.device_count()
        world_nodes = getattr(args, "world_size", 1)
        global_batch = getattr(args, "_global_batch_per_step", args.train.batch_size)
        print(f"[CFG] gpus_per_node={gpus_per_node} nodes={world_nodes} global_batch={global_batch} "
              f"base_lr={base_lr:.3e} warmup_lr={warmup_lr:.3e} final_lr={final_lr:.3e}")

    # GPU augmenter
    gpu_view = GPUViewGen().to(device)
    gpu_view.train()

    # logger
    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)

    # profiler (rank-0)
    prof = maybe_create_profiler(args) if is_main_process() else None

    # resume checkpoint
    start_epoch = resume_if_needed(args, model, optimizer, lr_scheduler)

    # progress
    global_progress = tqdm(range(start_epoch, args.train.stop_at_epoch), desc='Training') if is_main_process() \
                      else range(start_epoch, args.train.stop_at_epoch)

    data_time_meter = SmoothedValue(); fwd_bwd_time_meter = SmoothedValue(); step_time_meter = SmoothedValue()
    best_acc = float('-inf'); best_path = None; last_epoch = start_epoch - 1
    accuracy = 0

    for epoch in global_progress:
        last_epoch = epoch

        # DDP epoch seed
        if getattr(args, 'distributed', False):
            from torch.utils.data.distributed import DistributedSampler
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

        model.train()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        elif hasattr(getattr(train_loader.dataset, "dataset", None), "set_epoch"):
            train_loader.dataset.dataset.set_epoch(epoch)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        local_iter = tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=not is_main_process()) \
                     if is_main_process() else train_loader

        prev = time.time()
        for idx, batch in enumerate(local_iter):
            if idx >= steps_per_epoch:
                break
            t0 = time.time()
            if getattr(args, "limit_train_steps", 0) and idx >= args.limit_train_steps:
                break

            # ----- forward (two branches: SimCLR-like or mixup-dict) -----
            if isinstance(batch, tuple):
                ((x1, x2), labels) = batch
                x1 = x1.to(device, non_blocking=True); x2 = x2.to(device, non_blocking=True)
                x1_aug = gpu_view(x1); x2_aug = gpu_view(x2)
                x1_aug = x1_aug.contiguous(memory_format=torch.channels_last)
                x2_aug = x2_aug.contiguous(memory_format=torch.channels_last)
            else:
                batch, _ = batch
                xi = batch['xi'].to(device, non_blocking=True)
                xj = batch['xj'].to(device, non_blocking=True)
                lam = batch['lam'].to(device, non_blocking=True)
                xi_aug = gpu_view(xi); xj_aug = gpu_view(xj)
                if lam.dim() == 1:
                    lam = lam.view(-1, 1, 1, 1)
                elif lam.dim() == 3:
                    lam = lam.unsqueeze(1)
                x1_aug = xi_aug
                x2_aug = xi_aug * lam + xj_aug * (1 - lam)
                x1_aug = x1_aug.contiguous(memory_format=torch.channels_last)
                x2_aug = x2_aug.contiguous(memory_format=torch.channels_last)

            # loss
            model.zero_grad(set_to_none=True)
            data_dict = model(x1_aug, x2_aug)
            loss = data_dict['loss'].mean() * _weight_const_for_rank(args)
            t1 = time.time()

            loss.backward(); optimizer.step(); lr_scheduler.step()
            data_dict.update({'lr': lr_scheduler.get_lr()})
            t2 = time.time()

            # profiler step (rank-0)
            if prof is not None:
                prof.step()

            # meters
            if getattr(args, "debug_profile", 0) in (1,2):
                data_time_meter.update(t1 - t0)
                fwd_bwd_time_meter.update(t2 - t1)
                step_time_meter.update(t2 - prev)
            prev = t2

            if idx == 0 and is_main_process() and torch.cuda.is_available():
                torch.cuda.synchronize()
                print("[Warmup] max_memory_allocated:", format_mem(torch.cuda.max_memory_allocated(device)))

            if is_main_process():
                safe_log = {}
                for k, v in data_dict.items():
                    try:
                        if isinstance(v, (int, float)):
                            safe_log[k] = float(v)
                        elif torch.is_tensor(v) and v.numel() == 1:
                            safe_log[k] = v.item()
                        else:
                            safe_log[k] = v
                    except Exception:
                        safe_log[k] = v
                local_iter.set_postfix(safe_log)
                logger.update_scalers(safe_log)

        # kNN monitor
        ran_monitor = False
        if is_main_process() and args.train.knn_monitor and (epoch % args.train.knn_interval == 0):
            accuracy = knn_monitor(
                backbone_for_eval, memory_loader, test_loader, device,
                k=min(args.train.knn_k, len(memory_loader.dataset)),
                hide_progress=args.hide_progress,
            )
            ran_monitor = True

        if is_main_process():
            epoch_dict = {"epoch": epoch, "accuracy": accuracy}
            if isinstance(global_progress, tqdm.__class__):
                global_progress.set_postfix(epoch_dict)
            logger.update_scalers(epoch_dict)

        if is_main_process() and getattr(args, "debug_profile", 0) in (1,2):
            peak_mem = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
            print(
                f"[Epoch {epoch}] data_time(avg): {data_time_meter.avg:.4f}s | "
                f"fwd_bwd(avg): {fwd_bwd_time_meter.avg:.4f}s | "
                f"step_time(avg): {step_time_meter.avg:.4f}s | "
                f"peak_mem: {format_mem(peak_mem)}"
            )

        # save per-epoch & best
        if is_main_process():
            save_epoch_checkpoint(args, model, optimizer, lr_scheduler, epoch)
            if ran_monitor and (accuracy >= best_acc):
                best_acc = accuracy
                best_path = save_best_checkpoint(args, model, optimizer, lr_scheduler, epoch, best_acc)

    # finalize profiler
    if prof is not None:
        prof.stop()

    # rank-0 final save + eval_from
    if is_main_process():
        os.makedirs(args.ckpt_dir, exist_ok=True)
        model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth")
        save_final(args, model, last_epoch, model_path)
        print(f"Model saved to {model_path}")
        with open(os.path.join(args.log_dir, "checkpoint_path.txt"), 'w+') as f:
            f.write(f'{best_path if best_path is not None else model_path}')
        args.eval_from = best_path if best_path is not None else model_path

    # close logger
    if is_main_process():
        try:
            if hasattr(logger, "close"):
                logger.close()
            elif hasattr(logger, "writer") and logger.writer is not None:
                logger.writer.close()
        except Exception as e:
            print(f"logger close err: {e}")
            

def run_linear_eval_and_finalize(args):
    # run linear eval (expects args.eval_from set by trainer)
    linear_eval(args)
    # rename log dir
    try:
        completed_log_dir = args.log_dir.replace('in-progress', 'debug' if getattr(args, "debug", False) else 'completed')
        os.rename(args.log_dir, completed_log_dir)
        print(f'Log file has been saved to {completed_log_dir}')
    except Exception as e:
        print(f'Err rename log dir: {e}')