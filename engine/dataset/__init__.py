# import torch
# import torchvision
# from .random_dataset import RandomDataset


# def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None):
#     if dataset == 'mnist':
#         dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
#     elif dataset == 'stl10':
#         dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
#     elif dataset == 'cifar10':
#         dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
#     elif dataset == 'cifar100':
#         dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
#     elif dataset == 'imagenet':
#         dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
#     elif dataset == 'random':
#         dataset = RandomDataset()
#     else:
#         raise NotImplementedError

#     if debug_subset_size is not None:
#         dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
#         dataset.classes = dataset.dataset.classes
#         dataset.targets = dataset.dataset.targets
#     return dataset

import torch
import torchvision
from .random_dataset import RandomDataset

def get_dataset(
    dataset,
    data_dir=None,
    transform=None,
    train=True,
    download=False,
    debug_subset_size=None,
    **kwargs,                # <— nhận thêm tham số tuỳ chọn
):
    """
    Các kwargs dành cho CSV mode:
      - image_dir: str, thư mục gốc ảnh
      - csv_path_train: str, CSV train
      - csv_path_val: str, CSV val (dùng khi train=False)
      - mixup_alpha: float
      - mixup_same_class: bool
      - use_schedule: bool
      - total_epochs: int
      - mixup_steps: int
      - mixup_lam_min: float
      - mixup_lam_max: float
      - mixup_alpha_min: float
      - mixup_alpha_max: float
    """
    if dataset == 'mnist':
        ds = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)

    elif dataset == 'stl10':
        ds = torchvision.datasets.STL10(
            data_dir, split='train+unlabeled' if train else 'test',
            transform=transform, download=download
        )

    elif dataset == 'cifar10':
        ds = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)

    elif dataset == 'cifar100':
        ds = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)

    elif dataset == 'imagenet':
        ds = torchvision.datasets.ImageNet(
            data_dir, split=('train' if train else 'val'),
            transform=transform, download=download
        )

    elif dataset == 'random':
        ds = RandomDataset()

    # ====== CSV datasets ======
    elif dataset == 'endo_mixup':
        # train dataset tạo [x1, x2] với mixup + (optional) schedule
        from .endo_dataset import EndoBalancedMixupDataset, StepLambdaScheduler
        # print(kwargs)
        image_dir = kwargs.get('image_dir', './data')
        csv_path = kwargs.get('train_csv') if train else kwargs.get('val_csv', kwargs.get('train_csv'))
        if csv_path is None:
            raise ValueError("Vui lòng truyền csv_path_train (và csv_path_val nếu cần) trong kwargs cho 'endo_mixup'.")

        scheduler = None
        if kwargs.get('use_schedule', False):
            scheduler = StepLambdaScheduler(
                total_epochs=kwargs['total_epochs'],
                steps=kwargs.get('mixup_steps', 4),
                lam_min=kwargs.get('mixup_lam_min', 0.0),
                lam_max=kwargs.get('mixup_lam_max', 0.5),
                alpha_min=kwargs.get('mixup_alpha_min', 1e-4),
                alpha_max=kwargs.get('mixup_alpha_max', 0.2),
            )

        ds = EndoBalancedMixupDataset(
            csv_path=csv_path,
            root_dir=image_dir,
            base_transform=transform,
            alpha=kwargs.get('mixup_alpha', 0.4),
            same_class=kwargs.get('mixup_same_class', True),
            scheduler=scheduler
        )

    elif dataset == 'endo_labeled':
        # memory/test dataset có nhãn (phục vụ kNN monitor)
        from .endo_dataset import CSVDatasetWithLabel
        image_dir = kwargs.get('image_dir', './data')
        csv_path = kwargs.get('train_csv') if train else kwargs.get('val_csv', kwargs.get('train_csv'))
        if csv_path is None:
            raise ValueError("Vui lòng truyền csv_path_train (và csv_path_val nếu cần) trong kwargs cho 'endo_mixup'.")

        ds = CSVDatasetWithLabel(
            csv_path=csv_path,
            root_dir=image_dir,
            transform=transform
        )

    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")

    # ====== debug subset giữ an toàn ======
    if debug_subset_size is not None:
        ds = torch.utils.data.Subset(ds, range(0, debug_subset_size))
        # cố gắng gắn .classes/.targets nếu tồn tại ở dataset gốc
        base = getattr(ds, 'dataset', None)
        if base is not None:
            if hasattr(base, 'classes'):
                ds.classes = base.classes
            if hasattr(base, 'targets'):
                ds.targets = base.targets

    return ds