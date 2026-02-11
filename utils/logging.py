import os, sys
import time

import wandb
from torch.utils.tensorboard import SummaryWriter
import pickle

class Logging:
    def __init__(self, args):
        self.__log = {}
        self.__epoch = 0

        if args.wandb:
            args.run_name = f"{args.ds}_{args.task}_{args.loss}_{args.model}__{int(time.time())}"

            self.__run = wandb.init(
                project=args.wandb_prj,
                entity=args.wandb_entity,
                config=args,
                name=args.run_name,
                force=True
            )

        if args.log:
            self.__writer = SummaryWriter(args.exp_dir)
            self.__writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )
        
        self.__args = args

    
    def __call__(self, key, value):
        if key in self.__log:
            self.__log[key] += value
        else:
            self.__log[key] = value
    
    def __update_wandb(self):
        for log_key in self.__log_avg:
            self.__run.log({log_key: self.__log_avg[log_key]}, step=self.__epoch)
    
    def __update_board(self):
        for log_key in self.__log_avg:
            self.__writer.add_scalar(log_key, self.__log_avg[log_key], self.__epoch)
    
    def __reset_epoch(self):
        self.__log = {}
    
    def reset(self):
        self.__reset_epoch()
        self.__epoch = 0
    
    def step(self, epoch):
        self.__epoch = epoch
        
        self.__log_avg = {}
        for log_key in self.__log:
            if log_key.split("/")[-1] in ['loss', 'depth_abs_error', 'depth_rel_error', 'iou']:
                if 'train' in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_train_batch
                elif 'valid' in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_valid_batch
                elif 'test' in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_test_batch
                else:
                    raise ValueError(f'key: {log_key} wrong format')
            else:
                if 'train' in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_train_sample
                elif 'valid' in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_valid_sample
                elif 'test' in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_test_sample
                else:
                    raise ValueError(f'key: {log_key} wrong format')

        if self.__args.wandb:
            self.__update_wandb()
        
        if self.__args.log:
            self.__update_board()

        self.__reset_epoch()
    
    def watch(self, model):
        self.__run.watch(models=model, log='all', log_freq=self.__args.num_train_batch, log_graph=True)
    
    def log_model(self):
        best_path = self.__args.exp_dir + f"/best.pt"
        if os.path.exists(best_path):
            self.__run.log_model(path=best_path, model_name=f"{self.__args.run_name}-best-model")
        last_path = self.__args.exp_dir + f"/last.pt"
        if os.path.exists(last_path):
            self.__run.log_model(path=last_path, model_name=f"{self.__args.run_name}-last-model")
    
    @property
    def log(self):
        return self._Logging__log
    
    @property
    def log_avg(self):
        return self._Logging__log_avg
    
    @property
    def epoch(self):
        return self._Logging__epoch

    @property
    def args(self):
        return self._Logging__args