import os, sys
import argparse
from typing import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='IBLA')

    # DATASET
    parser.add_argument('--ds', type=str, required=True, choices = [
        'oxford', 'nyu', 'celeb', 'city', 'cifar10lt', 'cifar100lt', 'cifar10', 'cifar100','vocalfolds', 'busi'
        ],
        help='dataset used in training')
    parser.add_argument('--bs', type=int, required=True, default=64,
        help='batch size used for data set')
    parser.add_argument('--pinmem', action='store_true',
        help='toggle to pin memory in data loader')
    parser.add_argument('--wk', type=int, default=12,  
        help='number of worker processor contributing to data preprocessing')
    parser.add_argument('--citi_mode', type=str, default='fine',  choices=['fine', 'coarse'],
        help='mode used for cityscape dataset')
    
    # DATASET - CIFAR10/100 LT
    parser.add_argument('--imb_type', type=str, default='exp', choices=['exp', 'step'],
        help='type of imbalance')
    parser.add_argument('--imb_factor', type=float, default=0.01, #0.1, 0.01, 0.05, 0.001
        help='imbalance factor')
    parser.add_argument('--rand_number', type=int, default=0,
        help='seed random number')
    
    # TRAINING GENERAL SETTINGS
    parser.add_argument('--idx', type=int, default=0,
        help='device index used in training')
    parser.add_argument('--seed', type=int, default=0,
        help='seed used in training')
    parser.add_argument('--model', type=str, default='unet', choices=[
        'unet', 'segnet', 'runet', 'attunet', 'rattunet', 'nestunet',
        'resnet18', 'base', 'resnet18_scratch'],
        help='backbone used in training')
    parser.add_argument('--loss', type=str, default='vanilla', 
        choices=['vanilla', 'focal', 'cb', 'cbfocal', 'bsl', 'gumfocal', 'gum', 'cag', 'na', 'ina', 'ldam', 'blv', 'inap'],
        help='loss function used in training')
    parser.add_argument('--task', type=str, default='clf', required=True,
        choices=['clf', 'seg'],
        help='training task')
    parser.add_argument('--epochs', type=int, default=100,
        help='number of epochs used in training')
    parser.add_argument('--test', action='store_true',
        help='toggle to say that this experiment is just flow testing')

    # LOGGING
    parser.add_argument('--wandb', action='store_true',
        help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true',
        help='toggle to use tensorboard for offline saving')
    parser.add_argument('--wandb_prj', type=str, default="IBLA",
        help='toggle to use wandb for online saving')
    parser.add_argument('--wandb_entity', type=str, default="scalemind",
        help='toggle to use wandb for online saving')

    # MODEL
    parser.add_argument('--init_ch', type=int, default=32,
        help='number of kernel in the first')
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate')
    
    # FOCAL - CLASS-BALANCE FOCAL LOSS - GUMBEL FOCAL LOSS
    parser.add_argument('--gamma', type=float, default=0, 
        help="gamma hyperparameter used in focal loss")
    
    # GUMBEL - GUMBEL FOCAL LOSS
    parser.add_argument('--gumbel_tau', type=float, default=1, 
        help="temperature factor used for gumbel softmax")
    parser.add_argument('--gumbel_hard', type=bool, default=True, 
        help="toggle to use reparameterizaiton trick in gumbel softmax")

    # CAG
    parser.add_argument('--cagrad_c', type=float, default=0.5,
        help='scale parameter in cag loss')

    # NA
    parser.add_argument('--na_alpha', type=float, default=-0.5,
        help='temp in noise adaptive loss')
    
    # LDAM
    parser.add_argument('--ldam_max_m', type=float, default=0.5,
        help='max_m param')
    parser.add_argument('--ldam_s', type=float, default=30,
        help='scale param')

    # BLV
    parser.add_argument('--blv_s', type=float, default=1,
        help='sigma param')


    args = parser.parse_args()

    from trainer import train_func

    train_func(args)