import argparse
import os
import torch

import numpy as np
import torch
import random

import re 
import yaml

import shutil
import warnings

from datetime import datetime


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value
    
    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")

def _ns_to_dict(ns):
    if isinstance(ns, Namespace):
        return {k: _ns_to_dict(v) for k, v in ns.__dict__.items()}
    return ns

def _parse_override(overrides):
    """Parse list of key=value to dict (flat)."""
    d = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}")
        k, v = item.split("=", 1)
        try:
            v = yaml.safe_load(v)   # ép kiểu số/bool
        except Exception:
            pass
        d[k.strip()] = v
    return d

def _update_dict(base: dict, updates: dict):
    for k, v in updates.items():
        keys = k.split(".")
        cur = base
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        cur[keys[-1]] = v

        
def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

def create_unique_folder(base_folder_path):
    if not os.path.exists(base_folder_path):
        os.makedirs(base_folder_path)
        print(f"Đã tạo thư mục: {base_folder_path}")
        return base_folder_path
    
    count = 1
    while True:
        new_folder_path = f"{base_folder_path}_{count}"
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"Created folder: {new_folder_path}")
            return new_folder_path
        count += 1
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    parser.add_argument("--cli-override", nargs="*", default=[], help="key=value (dotted path). Ví dụ: model.backbone=resnet50")
    
    parser.add_argument("--use_csv", action="store_true")
    parser.add_argument("--image_dir", type=str, default="./data")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument('--train_binary', action='store_true')


    # ... trong get_args() sau các add_argument hiện có:
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world-size', default=1, type=int,
                        help='Number of nodes (machines) participating')
    parser.add_argument('--rank', default=0, type=int,
                        help='Node rank in [0..world_size-1]')
    parser.add_argument('--local_rank', type=int, default=0)  # torchrun set ENV, nhưng có default cũng tốt
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug_profile', type=int, default=0,
                    help='1: bật đo thời gian + memory; 2: thêm torch.profiler 30 step đầu')
    parser.add_argument('--limit_train_steps', type=int, default=0,
                        help='Giới hạn số batch mỗi epoch khi debug (0 = không giới hạn)')
    # arguments.py
    parser.add_argument('--resume-self-supervised', type=str, default=None,
                        help='Path tới checkpoint .pth để resume pretraining')
    parser.add_argument('--resume-eval-from', type=str, default=None,
                        help='Path tới checkpoint .pth để resume eval')

    parser.add_argument('--per-gpu-batch', type=str, default='',
                        help='Khai báo per-GPU batch theo index local: "0:8,1:12,2:6,3:6". '
                             'Ưu tiên dùng khi VRAM lệch.')

    # tiện: tắt tqdm & log ở non-master
    parser.add_argument('--print-freq', default=50, type=int)

    # Mixup / schedule
    parser.add_argument("--no_mixup_same_class", action="store_true", help="Disable same-class pairing (default: turn on)")
    parser.add_argument("--mixup_steps", type=int, default=4)
    parser.add_argument("--mixup_lam_min", type=float, default=0.0)
    parser.add_argument("--mixup_lam_max", type=float, default=0.5)
    parser.add_argument("--mixup_alpha_min", type=float, default=1e-4)
    parser.add_argument("--mixup_alpha_max", type=float, default=0.2)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=8)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
    parser.add_argument('--log_dir', type=str, default=os.getenv('LOG'))
    parser.add_argument('--ckpt_dir', type=str, default=os.getenv('CHECKPOINT'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    args = parser.parse_args()
    args.mixup_same_class = not args.no_mixup_same_class

    with open(args.config_file, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if args.cli_override:
        # ưu tiên CLI override
        overrides = _parse_override(args.cli_override)
        _update_dict(cfg, overrides)

    # merge config (sau override) vào args
    merged = Namespace(cfg)
    for k, v in merged.__dict__.items():
        vars(args)[k] = v
        
    # with open(args.config_file, 'r') as f:
    #     for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
    #         vars(args)[key] = value

    if args.debug:
        if args.train: 
            args.train.batch_size = 2
            args.train.num_epochs = 1
            args.train.stop_at_epoch = 1
        if args.eval: 
            args.eval.batch_size = 2
            args.eval.num_epochs = 1 # train only one epoch
        args.dataset.num_workers = 0


    ## make log dir & ckpt dir
    
    if args.log_dir is None:
        args.log_dir = "runs/"
    
    if args.resume_self_supervised:
        resume_path = args.resume_self_supervised
    if args.resume_eval_from:
        resume_path = args.resume_eval_from
    if resume_path is not None:
        resume_path = os.path.expanduser(resume_path)
        # Cho phép truyền vào: (1) file .pth, (2) thư mục checkpoints/, (3) chính log_dir
        if os.path.isfile(resume_path):
            cand = os.path.dirname(resume_path)           # .../checkpoints
        else:
            cand = resume_path                            # có thể là .../checkpoints hoặc .../logdir

        base = os.path.basename(os.path.normpath(cand))
        if base == "checkpoints":
            log_dir = os.path.dirname(cand)               # lấy thư mục cha của checkpoints
            ckpt_dir = cand
        else:
            log_dir = cand
            ckpt_dir = os.path.join(log_dir, "checkpoints")
        
        args.log_dir = log_dir
        args.ckpt_dir = ckpt_dir
    
    else:
        if args.model.pretrained_backbone:
            pretrained_tag = "with_pretrained"
        else:
            pretrained_tag = "no_pretrained"
        if args.train_binary:
            binary_tag = "_binary"
        else:
            binary_tag = ""
        if args.set_name:
            set_tag = "_" + args.set_name
        else:
            set_tag = ""
        log_dir = os.path.join(args.log_dir, args.model.name + "_" + args.model.aug_name,
            args.model.backbone+'_b'+ str(args.train.batch_size)+'_p'+ str(args.model.proj_layers) + '_' + pretrained_tag + binary_tag + set_tag)
        args.log_dir = create_unique_folder(log_dir)
        args.ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    
    print(f'creating file {args.log_dir}')
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)


    # ======= ++++++++++++++++++++++++++++++++++++++++++++++++++++ =======
    merged_cfg_path = os.path.join(args.log_dir, "config.yaml")
    # (tuỳ chọn) nhét log_dir/ckpt_dir vào cfg để downstream biết
    cfg["log_dir"] = args.log_dir
    cfg["ckpt_dir"] = args.ckpt_dir
    with open(merged_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"✓ Saved merged config to: {merged_cfg_path}")
    # ---------------------------------------------------------------------

    # shutil.copy2(args.config_file, args.log_dir)
    set_deterministic(args.seed)


    vars(args)['aug_kwargs'] = {
        # 'name':args.model.name,
        'aug_name': args.model.aug_name,
        'image_size': args.dataset.image_size
    }
    vars(args)['dataset_kwargs'] = {
        'dataset':args.dataset.name,
        'data_dir': args.data_dir,
        'download':args.download,
        'debug_subset_size': args.debug_subset_size if args.debug else None,
    }
    
    # >>> THÊM KHỐI NÀY <<<
    if args.use_csv:
        vars(args)['dataset_kwargs'].update({
            'image_dir': args.image_dir,
            'train_csv': args.train_csv,
            'val_csv': args.val_csv,
            # schedule / mixup config
            'use_schedule': True,
            'total_epochs': args.train.num_epochs,
            'mixup_steps': args.mixup_steps,
            'mixup_lam_min': args.mixup_lam_min,
            'mixup_lam_max': args.mixup_lam_max,
            'mixup_alpha_min': args.mixup_alpha_min,
            'mixup_alpha_max': args.mixup_alpha_max,
            'mixup_alpha': args.mixup_alpha,
            'mixup_same_class': args.mixup_same_class,
        })
    
    vars(args)['dataloader_kwargs'] = {
        # 'drop_last': True,
        # 'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    return args

if __name__ == "__main__":
    args = get_args()
    print("Arguments:") 
    for key, value in vars(args).items(): 
        print(f"{key}: {value}")