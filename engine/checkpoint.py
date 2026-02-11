import os
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP


def _unwrap(model):
    return model.module if isinstance(model, (DDP, DataParallel)) else model


def resume_if_needed(args, model, optimizer=None, lr_scheduler=None):
    start_epoch = 0
    if getattr(args, 'resume_self_supervised', None):
        ckpt = torch.load(args.resume_self_supervised, map_location='cpu')
        target = _unwrap(model)
        missing, unexpected = target.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"[Resume] missing={missing}, unexpected={unexpected}")
        if optimizer is not None and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if lr_scheduler is not None and 'lr_scheduler' in ckpt:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        start_epoch = int(ckpt.get('epoch', 0))
    return start_epoch


def save_epoch_checkpoint(args, model, optimizer, lr_scheduler, epoch):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    to_save = _unwrap(model)
    epoch_path = os.path.join(args.ckpt_dir, f"{args.name}_epoch{epoch:03d}.pth")
    torch.save({
        'epoch': epoch + 1,
        'state_dict': to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, epoch_path)
    return epoch_path


def save_best_checkpoint(args, model, optimizer, lr_scheduler, epoch, best_acc):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    to_save = _unwrap(model)
    best_path = os.path.join(args.ckpt_dir, f"{args.name}_best.pth")
    torch.save({
        'epoch': epoch + 1,
        'state_dict': to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_acc': best_acc,
    }, best_path)
    return best_path


def save_final(args, model, last_epoch, model_path):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    to_save = _unwrap(model)
    torch.save({'epoch': last_epoch + 1, 'state_dict': to_save.state_dict()}, model_path)
