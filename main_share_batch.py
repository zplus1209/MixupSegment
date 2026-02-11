import os
import time
import random
import numpy as np

import kornia.augmentation as K
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datetime import datetime

from arguments import get_args
from augmentations import get_aug
from engine.models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from engine.dataset import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from engine.linear_eval import main as linear_eval
from engine.dataset.utils import classes, CUDAPrefetcher

# ---------- utils ----------
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

def setup_for_distributed(is_master: bool):
    """Chỉ in log ở tiến trình master."""
    import builtins as __builtins__
    builtin_print = __builtins__.print
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtins__.print = print

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

class GPUViewGen(torch.nn.Module):
    """Sinh 1 view augment NGẪU NHIÊN trên GPU + Normalize"""
    def __init__(self):
        super().__init__()
        self.augs = torch.nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            K.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, p=0.8),
            K.Normalize(mean=imagenet_mean, std=imagenet_std),
        )

    @torch.no_grad()
    def forward(self, x):
        x = x.contiguous(memory_format=torch.contiguous_format)
        return self.augs(x)
    
class SmoothedValue:
    def __init__(self):
        self.n = 0
        self.total = 0.0
    def update(self, val, k=1):
        self.total += float(val) * k
        self.n += k
    @property
    def avg(self):
        return self.total / max(1, self.n)

def format_mem(bytes_):
    gb = bytes_ / (1024**3)
    return f"{gb:.2f} GB"

def parse_per_gpu_batch(spec: str):
    """
    spec dạng "0:8,1:12,2:6" -> dict {0:8, 1:12, 2:6}
    """
    mapping = {}
    if not spec:
        return mapping
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(':')
        mapping[int(k)] = int(v)
    return mapping

def get_total_mem_mb(gpu_idx: int) -> int:
    props = torch.cuda.get_device_properties(gpu_idx)
    return int(props.total_memory // (1024 * 1024))

def compute_local_batch(args, ngpus_per_node: int):
    local_idx = args.local_rank
    mapping = parse_per_gpu_batch(getattr(args, 'per_gpu_batch', ''))
    if local_idx in mapping:
        return max(1, int(mapping[local_idx]))
    return int(args.train.batch_size)

def filter_for_pair_requirement(samples):
    from collections import Counter, defaultdict
    by_cls = defaultdict(list)
    for i, s in enumerate(samples):
        by_cls[int(s["label"])].append(i)
    keep = set()
    for c, idxs in by_cls.items():
        if len(idxs) >= 2:
            keep.update(idxs)
    return [samples[i] for i in sorted(keep)]

# ---------- training main ----------
def main(device, args):
    if not hasattr(args, "debug_profile"):
        args.debug_profile = 0
    if not hasattr(args, "limit_train_steps"):
        args.limit_train_steps = 0

    train_base_tf = get_aug(train=True, **args.aug_kwargs)
    eval_base_tf  = get_aug(train=False, train_classifier=False, **args.aug_kwargs)

    dataset_kwargs = dict(args.dataset_kwargs)
    dataset_kwargs.pop('dataset', None)

    # --- Datasets ---
    train_set = get_dataset(
        dataset=args.dataset.name,
        transform=train_base_tf,
        train=True,
        **dataset_kwargs
    )
    memory_set = get_dataset(
        dataset='endo_labeled',
        transform=eval_base_tf,
        train=True,
        **dataset_kwargs
    )
    test_set = get_dataset(
        dataset='endo_labeled',
        transform=eval_base_tf,
        train=False,
        **dataset_kwargs
    )

    # --- Sampler & Dataloaders ---
    if getattr(args, 'distributed', False):
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        shuffle_flag = False
    else:
        train_sampler = None
        shuffle_flag = True

    pw = args.dataloader_kwargs.get("num_workers", 0) > 0

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        sampler=train_sampler,
        shuffle=shuffle_flag,
        batch_size=(args._local_train_batch if getattr(args, "_local_train_batch", None) else args.train.batch_size),
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=4 if pw else None,
        drop_last=True,
        **args.dataloader_kwargs
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=memory_set,
        shuffle=False,
        batch_size=args.train.batch_size,
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=2 if pw else None,
        drop_last=False,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        shuffle=False,
        batch_size=args.train.batch_size,
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=2 if pw else None,
        drop_last=False,
        **args.dataloader_kwargs
    )

    # Đồng bộ steps/epoch
    local_steps = torch.tensor([len(train_loader)], dtype=torch.int64, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(local_steps, op=dist.ReduceOp.MIN)
    steps_per_epoch = int(local_steps.item())

    if is_main_process():
        print(f"[StepsPerEpoch] steps_per_epoch={steps_per_epoch} (min over ranks)")

    # --- Model ---
    model = get_model(args.model).to(device)

    use_ddp = getattr(args, 'distributed', False)
    if use_ddp:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
        backbone_for_eval = model.module.backbone
    else:
        model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        backbone_for_eval = model.module.backbone if isinstance(model, torch.nn.DataParallel) else model.backbone

    model = model.to(memory_format=torch.channels_last)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = model.to(memory_format=torch.channels_last)

    # tạo GPU augmenter
    gpu_view = GPUViewGen().to(device)
    gpu_view.train()
    
    # --- Optimizer & Scheduler (scale theo global batch) ---
    if use_ddp:
        global_batch = getattr(args, "_global_batch_per_step", None)
        if global_batch is None:
            global_batch = args.train.batch_size * torch.cuda.device_count() * getattr(args, "world_size", 1)
    else:
        global_batch = args.train.batch_size

    lr_scale = global_batch / 256.0
    base_lr   = args.train.base_lr   * lr_scale
    final_lr  = args.train.final_lr  * lr_scale
    warmup_lr = args.train.warmup_lr * lr_scale
    
    gpus_per_node = torch.cuda.device_count()
    world_nodes = getattr(args, "world_size", 1)

    if is_main_process():
        print(f"[CFG] gpus_per_node={gpus_per_node} nodes={world_nodes} "
              f"global_batch={global_batch} "
              f"base_lr={base_lr:.3e} warmup_lr={warmup_lr:.3e} final_lr={final_lr:.3e}")

    optimizer = get_optimizer(
        args.train.optimizer.name, model,
        lr=base_lr,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay
    )

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, warmup_lr,
        args.train.num_epochs, base_lr, final_lr,
        steps_per_epoch,
        constant_predictor_lr=True
    )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)

    # --- Optional profiler ---
    use_profiler = (getattr(args, "debug_profile", 0) == 2)
    prof = None
    if use_profiler and is_main_process():
        from torch.profiler import profile, record_function, ProfilerActivity, schedule
        prof_sched = schedule(wait=5, warmup=5, active=20, repeat=1)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_sched,
            on_trace_ready=lambda p: p.export_chrome_trace(
                os.path.join(args.log_dir, f"trace_rank0_{datetime.now().strftime('%H%M%S')}.json")
            ),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        prof.start()

    # --- Resume checkpoint ---
    start_epoch = 0
    if getattr(args, 'resume_self_supervised', None):
        ckpt = torch.load(args.resume_self_supervised, map_location='cpu')
        target = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
        missing, unexpected = target.load_state_dict(ckpt['state_dict'], strict=False)
        if is_main_process():
            print(f"[Resume] missing={missing}, unexpected={unexpected}")
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'lr_scheduler' in ckpt:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        start_epoch = int(ckpt.get('epoch', 0))
        
    accuracy = 0
    global_progress = tqdm(range(start_epoch, args.train.stop_at_epoch), desc=f'Training') if is_main_process() \
                      else range(start_epoch, args.train.stop_at_epoch)

    measure = (getattr(args, "debug_profile", 0) in (1,2))
    data_time_meter = SmoothedValue()
    fwd_bwd_time_meter = SmoothedValue()
    step_time_meter = SmoothedValue()

    best_acc = float('-inf')
    best_path = None
    has_best = False

    last_epoch = start_epoch - 1   # ### CHANGED: theo dõi epoch cuối cùng (kể cả khi loop rỗng)
    
    for epoch in global_progress:
        last_epoch = epoch         # ### CHANGED: cập nhật mỗi vòng
        if use_ddp and isinstance(train_loader.sampler, DistributedSampler):
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

            # --------------- forward ---------------
            if isinstance(batch, tuple):
                ((x1, x2), labels) = batch
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                x1_aug = gpu_view(x1)
                x2_aug = gpu_view(x2)
                x1_aug = x1_aug.contiguous(memory_format=torch.channels_last)
                x2_aug = x2_aug.contiguous(memory_format=torch.channels_last)

            else:
                batch, _ = batch
                xi = batch['xi'].to(device, non_blocking=True)
                xj = batch['xj'].to(device, non_blocking=True)
                lam = batch['lam'].to(device, non_blocking=True)
                xi_aug = gpu_view(xi)
                xj_aug = gpu_view(xj)
                x1_aug = xi_aug
                if lam.dim() == 1:
                    lam = lam.view(-1, 1, 1, 1)
                elif lam.dim() == 3:
                    lam = lam.unsqueeze(1)
                x2_aug = xi_aug * lam + xj_aug * (1 - lam)
                x1_aug = x1_aug.contiguous(memory_format=torch.channels_last)
                x2_aug = x2_aug.contiguous(memory_format=torch.channels_last)

            model.zero_grad(set_to_none=True)
            data_dict = model(x1_aug, x2_aug)
            loss = data_dict['loss'].mean()

            if is_dist_avail_and_initialized():
                mean_bs = getattr(args, "_mean_bs", None)
                local_bs_const = getattr(args, "_local_train_batch", None)
                if mean_bs is None or local_bs_const is None:
                    local_bs_const = local_bs_const or args.train.batch_size
                    world_sz = dist.get_world_size()
                    gb = getattr(args, "_global_batch_per_step", local_bs_const * world_sz)
                    mean_bs = gb / float(world_sz)
                weight_const = float(local_bs_const) / max(1.0, float(mean_bs))
            else:
                weight_const = 1.0
            
            loss = loss * weight_const
            
            t1 = time.time()

            # --------------- backward ---------------
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr': lr_scheduler.get_lr()})

            t2 = time.time()

            if getattr(args, "debug_profile", 0) == 2 and is_main_process() and prof is not None:
                prof.step()

            if measure:
                data_time_meter.update(t1 - t0)
                fwd_bwd_time_meter.update(t2 - t1)
                step_time_meter.update(t2 - prev)
            prev = t2

            if idx == 0 and is_main_process() and torch.cuda.is_available():
                torch.cuda.synchronize()
                print("[Warmup] max_memory_allocated:",
                      format_mem(torch.cuda.max_memory_allocated(device)))

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
                hide_progress=args.hide_progress
            )
            ran_monitor = True

        if is_main_process():
            epoch_dict = {"epoch": epoch, "accuracy": accuracy}
            if isinstance(global_progress, tqdm.__class__):
                global_progress.set_postfix(epoch_dict)
            logger.update_scalers(epoch_dict)

        if is_main_process() and measure:
            peak_mem = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
            print(
                f"[Epoch {epoch}] data_time(avg): {data_time_meter.avg:.4f}s | "
                f"fwd_bwd(avg): {fwd_bwd_time_meter.avg:.4f}s | "
                f"step_time(avg): {step_time_meter.avg:.4f}s | "
                f"peak_mem: {format_mem(peak_mem)}"
            )
        
        if is_main_process():
            os.makedirs(args.ckpt_dir, exist_ok=True)
            to_save = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
            # lưu mỗi epoch
            epoch_path = os.path.join(args.ckpt_dir, f"{args.name}_epoch{epoch:03d}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, epoch_path)

            # chỉ lưu "best" khi có monitor
            if ran_monitor and (accuracy >= best_acc):   # ### CHANGED
                best_acc = accuracy
                best_path = os.path.join(args.ckpt_dir, f"{args.name}_best.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'best_acc': best_acc,
                }, best_path)

    if use_profiler and is_main_process() and prof is not None:
        prof.stop()

    # Save checkpoint cuối (master)
    if is_main_process():
        os.makedirs(args.ckpt_dir, exist_ok=True)
        save_epoch = last_epoch + 1                    # ### CHANGED
        model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth")
        to_save = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
        torch.save({'epoch': save_epoch, 'state_dict': to_save.state_dict()}, model_path)
        print(f"Model saved to {model_path}")
        with open(os.path.join(args.log_dir, "checkpoint_path.txt"), 'w+') as f:
            f.write(f'{model_path}')
        if best_path is not None:
            args.eval_from = best_path
        else:
            args.eval_from = model_path
        
    # đóng logger
    if is_main_process():
        try:
            if hasattr(logger, "close"):
                logger.close()
            elif hasattr(logger, "writer") and logger.writer is not None:
                logger.writer.close()
        except Exception as e:
            print(f"logger close err: {e}")
    return

# ---------- DDP worker ----------
def main_worker(gpu, ngpus_per_node, args):
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
        rank=args.rank
    )
    ngpus_per_node = torch.cuda.device_count()

    # Adaptive local batch
    local_batch = compute_local_batch(args, ngpus_per_node)

    # broadcast tổng batch/step
    local_bs_t = torch.tensor([local_batch], dtype=torch.int64, device=args.device)
    dist.all_reduce(local_bs_t, op=dist.ReduceOp.SUM)
    global_batch_per_step = int(local_bs_t.item())

    world_sz = dist.get_world_size()
    mean_bs = global_batch_per_step / float(world_sz)
    
    if is_main_process():
        print(f"[AdaptiveBatch] rank={args.rank} local_batch={local_batch} | "
              f"global_batch_per_step={global_batch_per_step}")

    args._local_train_batch = local_batch
    args._global_batch_per_step = global_batch_per_step
    args._mean_bs = mean_bs
    
    main(device=args.device, args=args)

    dist.barrier()
    dist.destroy_process_group()
    
    # rank-0 chạy linear eval
    if args.rank == 0:
        args.distributed = False
        args.multiprocessing_distributed = False

        if torch.cuda.is_available():
            torch.cuda.init()

        linear_eval(args)

        # đổi tên log dir
        try:
            completed_log_dir = args.log_dir.replace('in-progress', 'debug' if getattr(args, "debug", False) else 'completed')
            os.rename(args.log_dir, completed_log_dir)
            print(f'Log file has been saved to {completed_log_dir}')
        except Exception as e:
            print(f'Err rename log dir: {e}')
    else:
        return

# ---------- entry ----------
if __name__ == "__main__":
    args = get_args()

    if getattr(args, "seed", None) is None:
        args.seed = 42

    if not hasattr(args, "dist_backend"):
        args.dist_backend = "nccl"

    if not getattr(args, "multiprocessing_distributed", False):
        args.distributed = False
        args.device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        setup_for_distributed(True)
        print("Arguments:")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        set_seed(args.seed)
        torch.backends.cudnn.benchmark = True
        main(device=args.device, args=args)
        linear_eval(args)
        try:
            completed_log_dir = args.log_dir.replace('in-progress', 'debug' if getattr(args, "debug", False) else 'completed')
            os.rename(args.log_dir, completed_log_dir)
            print(f'Log file has been saved to {completed_log_dir}')
        except Exception as e:
            print(f'Err rename log dir: {e}')
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.world_size
        setup_for_distributed(is_master=True)
        print("Arguments:")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
