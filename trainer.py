import random
from rich.progress import track
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_ds
from utils import folder_setup, save_cfg, Logging, save_json, invnorm, invnorm255

from mapping import mapping


def train_func(args):

    # seed setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = args.idx)

    # folder setup and save setting
    args.exp_dir = folder_setup(args)
    save_cfg(args, args.exp_dir)

    # dataset setup
    data, args = get_ds(args)
    _, _, _, train_dl, valid_dl, _ = data

    # logging setup
    log_interface = Logging(args)

    # task mapping
    if args.task not in mapping[args.ds]:
        raise ValueError(f"Currently, task {args.task} is not supported")
    task_dict = mapping[args.ds][args.task]

    # metrics
    metric_dict = task_dict["metrics"]

    # loss
    train_loss_fn = task_dict["loss"][args.loss](args=args)
    eval_loss_fn = task_dict["loss"]['vanilla'](args=args)

    # model
    model = task_dict["model"][args.model](args=args).to(device)

    # optimizer, scheduler
    optimizer = Adam(model.parameters(), lr = 0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max= len(train_dl)*args.epochs)

    if args.wandb:
        log_interface.watch(model)

    # training
    old_valid_loss = 1e26

    for epoch in track(range(args.epochs)):
        args.epoch = epoch
        
        # train data loader
        model.train()
        for _, (img, target) in enumerate(train_dl):
            img = img.to(device)
            target = target.to(device)

            pred = model(img)
            loss = train_loss_fn(pred, target)

            log_interface(key="train/loss", value=loss.item())

            for metric_key in metric_dict:
                metric_value = metric_dict[metric_key](pred, target)
                log_interface(key=f"train/{metric_key}", value=metric_value)
            
            optimizer.zero_grad()
            if args.loss == 'cag':
                train_loss_fn.backward(list(model.parameters()))
            else:
                loss.backward()
            optimizer.step()
            scheduler.step()

        # valid data loader 
        model.eval()
        with torch.no_grad():
            for _, (img, target) in enumerate(valid_dl):
                img = img.to(device)
                target = target.to(device)

                pred = model(img)
                loss = eval_loss_fn(pred, target)

                log_interface(key="valid/loss", value=loss.item())

                for metric_key in metric_dict:
                    metric_value = metric_dict[metric_key](pred, target)
                    log_interface(key=f"valid/{metric_key}", value=metric_value)
        
        # Logging can averaging
        log_interface.step(epoch=epoch)

        # save best and last model
        mean_valid_loss = log_interface.log_avg["valid/loss"]
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_valid_loss
        }
        if  mean_valid_loss <= old_valid_loss:
            old_valid_loss = mean_valid_loss

            save_path = args.exp_dir + f"/best.pt"
            torch.save(save_dict, save_path)
        
        save_path = args.exp_dir + f"/last.pt"
        torch.save(save_dict, save_path)
    
    # save model
    log_interface.log_model()