import os, sys

from .nyu import NYUv2
from .cityscape import CustomCityScapeDS
from .celeb import CustomCeleb
from .ox import CustomOxFordPet
from .cifar import ImbalanceCIFAR10, ImbalanceCIFAR100

from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

from .vocalfolds import Customvocalfolds
from .busi import CustomBusi

def get_ds_busi(args):
    core_ds = CustomBusi(args=args, split='train')

    train_ds, valid_ds = random_split(core_ds, [0.9, 0.1])
    valid_ds.mode = 'test'
    test_ds = valid_ds

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)

    args.seg_n_classes = 1

    args.num_train_sample = len(train_ds)
    args.num_valid_sample = len(valid_ds)
    args.num_test_sample = len(test_ds)
    args.num_train_batch = len(train_dl)
    args.num_valid_batch = len(valid_dl)
    args.num_test_batch = len(test_dl)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl), args

def get_ds_ox(args):
    train_ds = CustomOxFordPet(split = 'trainval', args=args)
    test_ds = CustomOxFordPet(split = 'test', args=args)

    extra_train_ds, valid_ds, test_ds = random_split(test_ds, [0.8, 0.1, 0.1])
    valid_ds.mode = 'test'
    test_ds.mode = 'test'
    extra_train_ds.mode = 'train'

    train_ds = ConcatDataset([train_ds, extra_train_ds])

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)

    args.seg_n_classes = 3

    args.num_train_sample = len(train_ds)
    args.num_valid_sample = len(valid_ds)
    args.num_test_sample = len(test_ds)
    args.num_train_batch = len(train_dl)
    args.num_valid_batch = len(valid_dl)
    args.num_test_batch = len(test_dl)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl), args

def get_ds_nyu(args):
    ds = NYUv2(args=args)

    train_ds, valid_ds, test_ds = random_split(ds, [0.8, 0.1, 0.1])

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)

    args.seg_n_classes = 14

    args.num_train_sample = len(train_ds)
    args.num_valid_sample = len(valid_ds)
    args.num_test_sample = len(test_ds)
    args.num_train_batch = len(train_dl)
    args.num_valid_batch = len(valid_dl)
    args.num_test_batch = len(test_dl)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl), args

def get_ds_celeb(args):
    train_ds = CustomCeleb(split='train', args=args)
    valid_ds = CustomCeleb(split='valid', args=args)
    test_ds = CustomCeleb(split='test', args=args)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)

    args.num_train_sample = len(train_ds)
    args.num_valid_sample = len(valid_ds)
    args.num_test_sample = len(test_ds)
    args.num_train_batch = len(train_dl)
    args.num_valid_batch = len(valid_dl)
    args.num_test_batch = len(test_dl)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl), args

def get_ds_city(args):
    if args.citi_mode == 'fine':
        train_ds = CustomCityScapeDS(split='train', mode=args.citi_mode, args=args)
        valid_ds = CustomCityScapeDS(split='val', mode=args.citi_mode, args=args)
        test_ds = CustomCityScapeDS(split='test', mode=args.citi_mode, args=args)
    elif args.citi_mode == "coarse":
        train_ds_1 = CustomCityScapeDS(split='train', mode=args.citi_mode)
        train_ds_2 = CustomCityScapeDS(split='train_extra', mode=args.citi_mode)
        train_ds = ConcatDataset([train_ds_1, train_ds_2])
        valid_ds = CustomCityScapeDS(split='val', mode=args.citi_mode)
        test_ds = CustomCityScapeDS(split='val', mode=args.citi_mode)
    
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)

    args.seg_n_classes = 20

    args.num_train_sample = len(train_ds)
    args.num_valid_sample = len(valid_ds)
    args.num_test_sample = len(test_ds)
    args.num_train_batch = len(train_dl)
    args.num_valid_batch = len(valid_dl)
    args.num_test_batch = len(test_dl)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl), args

def get_ds_cifar_lt(args):
    if args.ds == 'cifar10lt':
        data_interface = ImbalanceCIFAR10
        args.n_classes = 10
    elif args.ds == 'cifar100lt':
        data_interface = ImbalanceCIFAR100
        args.n_classes = 100

    train_ds = data_interface(
        root="/".join(__file__.split("/")[:-1]) + "/source",
        args=args,
        train=True
    )

    valid_ds = data_interface(
        root="/".join(__file__.split("/")[:-1]) + "/source",
        args=args,
        train=False
    )

    test_ds = valid_ds

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)

    args.num_train_sample = len(train_ds)
    args.num_valid_sample = len(valid_ds)
    args.num_test_sample = len(test_ds)
    args.num_train_batch = len(train_dl)
    args.num_valid_batch = len(valid_dl)
    args.num_test_batch = len(test_dl)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl), args

def get_ds_cifar(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )
    
    data_interface = None
    
    if args.ds == 'cifar10':
        data_interface = CIFAR10
        args.n_classes = 10
    elif args.ds == 'cifar100':
        data_interface = CIFAR100
        args.n_classes = 100
    
    train_ds = data_interface(
        root="/".join(__file__.split("/")[:-1]) + "/source",
        transform=transform,
        download=True,
        train=True
    )
    
    valid_ds = data_interface(
        root="/".join(__file__.split("/")[:-1]) + "/source",
        transform=transform,
        download=True,
        train=False
    )

    test_ds = valid_ds
    
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)

    args.num_train_sample = len(train_ds)
    args.num_valid_sample = len(valid_ds)
    args.num_test_sample = len(test_ds)
    args.num_train_batch = len(train_dl)
    args.num_valid_batch = len(valid_dl)
    args.num_test_batch = len(test_dl)
    
    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl), args

def get_ds_vocalfolds(args):
    core_ds = Customvocalfolds(args=args, split='train')

    train_ds, valid_ds = random_split(core_ds, [0.9, 0.1])
    valid_ds.mode = 'test'
    test_ds = valid_ds

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)

    args.seg_n_classes = 7

    args.num_train_sample = len(train_ds)
    args.num_valid_sample = len(valid_ds)
    args.num_test_sample = len(test_ds)
    args.num_train_batch = len(train_dl)
    args.num_valid_batch = len(valid_dl)
    args.num_test_batch = len(test_dl)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl), args


def get_ds(args):

    ds_mapping = {
        "oxford" : get_ds_ox,
        "nyu" : get_ds_nyu,
        "celeb" : get_ds_celeb,
        "city" : get_ds_city,
        "cifar10lt" : get_ds_cifar_lt,
        "cifar100lt" : get_ds_cifar_lt,
        "cifar10" : get_ds_cifar,
        "cifar100" : get_ds_cifar,
        "vocalfolds" : get_ds_vocalfolds,
        "busi" : get_ds_busi
    }

    data, args = ds_mapping[args.ds](args)

    return data, args

