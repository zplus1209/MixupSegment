import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from engine.models import get_model
from optimizers import get_optimizer, LR_Scheduler


def build_model_optimizer_scheduler(args, steps_per_epoch, device):
    # model
    model = get_model(args.model).to(device)
    use_ddp = getattr(args, 'distributed', False)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        backbone_for_eval = model.module.backbone
    else:
        model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        backbone_for_eval = model.module.backbone if isinstance(model, torch.nn.DataParallel) else model.backbone

    # channels-last + TF32
    model = model.to(memory_format=torch.channels_last)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # lr scaling by global batch
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

    optimizer = get_optimizer(
        args.train.optimizer.name, model,
        lr=base_lr,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay,
    )

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, warmup_lr,
        args.train.num_epochs, base_lr, final_lr,
        steps_per_epoch,
        constant_predictor_lr=True,
    )

    return model, backbone_for_eval, optimizer, lr_scheduler, (base_lr, warmup_lr, final_lr)
