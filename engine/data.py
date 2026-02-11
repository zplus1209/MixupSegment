from typing import Tuple
import torch
from torch.utils.data.distributed import DistributedSampler

from augmentations import get_aug
from engine.dataset import get_dataset


def build_datasets_and_loaders(args, device) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    train_base_tf = get_aug(train=True, **args.aug_kwargs)
    eval_base_tf  = get_aug(train=False, train_classifier=False, **args.aug_kwargs)

    dataset_kwargs = dict(args.dataset_kwargs)
    dataset_kwargs.pop('dataset', None)

    # datasets
    train_set = get_dataset(dataset=args.dataset.name, transform=train_base_tf, train=True, **dataset_kwargs)
    memory_set = get_dataset(dataset='endo_labeled', transform=eval_base_tf, train=True, **dataset_kwargs)
    test_set = get_dataset(dataset='endo_labeled', transform=eval_base_tf, train=False, **dataset_kwargs)

    # sampler & loaders
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

    # min steps/epoch across ranks
    local_steps = torch.tensor([len(train_loader)], dtype=torch.int64, device=device)
    if args.distributed:
        import torch.distributed as dist
        dist.all_reduce(local_steps, op=dist.ReduceOp.MIN)
    steps_per_epoch = int(local_steps.item())

    return train_loader, memory_loader, test_loader, steps_per_epoch