import os
import math
import time
import datetime
import sys
import warnings

import numpy as np
from tqdm import tqdm
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from config import *
from config import Config as conf

from model import *
from data import *
from optimizer import *
from scheduler import *
from regularization import *
from pruning.pruning import *

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from hyperopt import hp
import copy
import pathlib


from utils.eval import Accumulator
from utils.meter import AverageMeter



warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "^(Palette)", UserWarning)



class Runner():
    def __init__(self, conf, ver_type=None, run_type=None, device_type=None):
        self.conf = conf

        self.ver_type = self.conf.get()['model']['ver']
        self.run_type = run_type
        self.device_type = device_type

        self.device = self.conf.get()['cuda']['device'] if device_type == None else device_type
        self.model = get_model(conf, self.ver_type, self.run_type, self.device_type)

        # { 'fp32' : full precision, 'fp16' : half precision }
        self.train_precision = 'fp16' if self.conf.get()['model']['name'] == 'micronet' else 'fp32'

        self.tr_loader, _ , self.test_loader = get_dataloader(conf)

        self.criterion = get_loss_fn(self.conf)
        self.optimizer = get_optimizer(self.model, self.conf)
        self.scheduler = get_scheduler(self.optimizer, self.conf)


        self.train_epochs = self.conf.get()['model']['training']['train_epochs']
        self.prune_rate=self.conf.get()['model']['pruning']['prune_rate']
        self.prune_epochs = self.conf.get()['model']['pruning']['prune_epochs']
        self.prune_iterations=self.conf.get()['model']['pruning']['prune_iterations']
        self.prune_target_ratio=self.conf.get()['model']['pruning']['prune_target_ratio']

        self.quantize_epochs = self.conf.get()['model']['quantization']['quantize_epochs']
        self.quantize_evals = 10

          
        self.best_acc_top1 = 0.0
        self.best_acc_top5 = 0.0
        self.best_test_loss = 9999.

        self.ebest_acc_top1 = 0.0
        self.ebest_acc_top5 = 0.0

    def model_print(self):
        print (self.model)

    def reset_average_meters(self):
        self.best_acc_top1 = 0.0
        self.best_acc_top5 = 0.0

        self.best_test_loss = 9999.
    

    def fn_checkpoint(self, method, suffix):
        if self.conf.get()['model']['pretrained']:
            trained = 'pretrained'
        else:
            trained = 'scratch'

        if method == 'train':
            model_fn =  "-tr-" + trained +"-lr"+ str(self.conf.get()['model']['lr']) + "-batch" +  \
                       str(self.conf.get()['model']['batch']) + "-epoch" +  \
                       str(self.conf.get()['model']['training']['train_epochs']) + "-" + str(suffix) +".pt"
        elif method == 'prune':
            model_fn =  "-pr-" + trained + "-lr"+ str(self.conf.get()['model']['lr']) + "-batch" + \
                       str(self.conf.get()['model']['batch']) + "-epoch" + \
                       str(self.conf.get()['model']['training']['train_epochs']) + "-" +str(suffix) +".pt"
        elif method == 'quantize':
            model_fn =  "-qt-" + trained + "-lr"+ str(self.conf.get()['model']['lr']) + "-batch" +  \
                       str(self.conf.get()['model']['batch']) + "-epoch" +  \
                       str(self.conf.get()['model']['training']['train_epochs']) + "-" +str(suffix) +".pt"
        else:
            raise ValueError('Unknown save_checkpoint() method')
        return model_fn

    def save_checkpoint(self,method, suffix):
        model_path = self.conf.get()['model']['path'] + "/"+  self.conf.get()['model']['name']

        if method == 'train':
            model_path += self.fn_checkpoint(method,suffix)
            torch.save({
                'model_state_dict': self.model.state_dict()
                }, model_path)
        elif method == 'prune':
            model_path += self.fn_checkpoint(method,suffix)
            torch.save({
                'model_state_dict': self.model.state_dict()
                }, model_path)
        elif method == 'quantize':
            model_path += self.fn_checkpoint(method,suffix)
            torch.jit.save(torch.jit.script(self.model), model_path )
        else:
            raise ValueError('Unknown save_checkpoint() method')


    def load_checkpoint(self,method, suffix):
        model_path = self.conf.get()['model']['path'] + "/"+  self.conf.get()['model']['name']

        if method == 'train':
            model_path += self.fn_checkpoint(method,suffix)
            checkpoint = torch.load(model_path)
            return checkpoint['model_state_dict']
        elif method == 'prune':
            model_path += self.fn_checkpoint(method,suffix)
            checkpoint = torch.load(model_path)
            return checkpoint['model_state_dict']
        elif method == 'quantize':
            model_path += self.fn_checkpoint(method,suffix)
            ckeckpoint = torch.jit.load(model_path)
            return checkpoint
        else:
            raise ValueError('Unknown save_checkpoint() method')

    def to_half(self):
        runner.model.half()
        for layer in runner.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
        self.train_precision = 'fp16'

    def to_full(self):
        runner.model.float()
        self.train_precision = 'fp32'


    def accuracy(self,output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(1. / batch_size))
        return res

    def regularizer(self,images, labels):
        if self.conf.get()['regularization']['name'] == 'cutout':
            lam, images, labels_a, labels_b = get_regularization(self.conf,images, labels, self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
        elif self.conf.get()['regularization']['name'] == 'mixup':
            lam, images, labels_a, labels_b = get_regularization(self.conf,images, labels, self.device)
        elif self.conf.get()['regularization']['name'] == 'cutmix':
            lam, images, labels_a, labels_b = get_regularization(self.conf, images, labels, self.device)
            outputs = self.model(images)
            loss = lam * self.criterion(outputs, labels_a) + (1-lam) * self.criterion(outputs, labels_b)
        elif self.conf.get()['regularization']['name'] == 'focusmix':
            lam, images, labels_a, labels_b = get_regularization(self.conf, images, labels, self.device)
            outputs = self.model(images)
            loss = lam * self.criterion(outputs, labels_a) + (1-lam) * self.criterion(outputs, labels_b)
        else:
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

        if self.conf.get()['regularization']['ortho']['avail']:
            args_ortho_lr = 0.7
            loss += args_ortho_lr * l2_reg_ortho_32bit(self.conf, self.model, self.device)

        return outputs, loss

    def evaluate(self, method, epoch):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        with torch.no_grad():
            if self.train_precision == 'fp16':
                self.model.half()
            for i, data in enumerate(tqdm(self.test_loader)):
                if conf.get()['data']['dali']['avail']:
                    timages = data[0]["data"]
                    tlabels = data[0]["label"].squeeze().long()
                else:
                    timages = data[0].type(torch.FloatTensor)
                    tlabels = data[1].type(torch.LongTensor)

                if self.conf.get()['cuda']['avail']:
                    timages, tlabels  = timages.to(self.device), tlabels.to(self.device) 
                    if self.train_precision == 'fp16':
                        timages = timages.half()
           
                toutputs = self.model(timages)

                tloss = self.criterion(toutputs, tlabels)

                ttop1, ttop5 = self.accuracy(toutputs, tlabels, (1, 5))

                losses.update(tloss.item(), timages.size(0))
                top1.update(ttop1.item(), timages.size(0))
                top5.update(ttop5.item(), timages.size(0))

            if self.best_acc_top1 < top1.avg:
                self.best_acc_top1 = top1.avg
                if self.best_acc_top5 < top5.avg:
                    self.best_acc_top5 = top5.avg
                self.save_checkpoint(method, "best")

            if self.best_test_loss > losses.avg:
                self.best_test_loss = losses.avg

            print('[{:d}/{:d}] <<<TEST>>> loss({:.4f}) top1({:.4f}) top5({:.4f}) best-top1({:.4f}) best-top5({:.4f})'.format(
                  epoch + 1, self.train_epochs, losses.avg, top1.avg, top5.avg, self.best_acc_top1, self.best_acc_top5))

def search(runner, conf_dir_file):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    def train_once(runner):
        for epoch in range(runner.train_epochs):
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            runner.model.train()

            for i, data in enumerate(runner.tr_loader):
                images, labels = data
                if conf.get()['cuda']['avail']:
                    images, labels  = images.to(runner.device), labels.to(runner.device)
                    runner.model = runner.model.to(runner.device)

                runner.optimizer.zero_grad()

                outputs, loss = runner.regularizer(images, labels)

                loss.backward()

                runner.optimizer.step()

                ttop1, ttop5 = runner.accuracy(outputs, labels, (1, 5))

                losses.update(loss.item(), images.size(0))
                top1.update(ttop1.item(), images.size(0))
                top5.update(ttop5.item(), images.size(0))

            print('[{:d}/{:d}] <<<TRAIN>>> lr({:.10f}) loss({:.4f}) top1({:.3f}) top5({:.3f})'.format(
                epoch + 1, runner.train_epochs, runner.optimizer.param_groups[0]['lr'], losses.avg, top1.avg, top5.avg)
            )
            runner.scheduler.step()
      

    def train(config):
        conf_dir_file= config['conf_dir_file']
        my_conf = Config(filename=conf_dir_file)
        
        for key, value in config.items():
            if key != 'conf_dir_file':
                my_conf.get()['model'][key] = value
        my_conf.get()['model']['name'] = 'rexnetv1_search'
        my_conf.get()['model']['input_ch'] = int(my_conf.get()['model']['input_ch'])
        my_conf.get()['model']['final_ch'] = int(my_conf.get()['model']['final_ch'])
        my_conf.get()['model']['use_se'] = round(my_conf.get()['model']['use_se'])
        my_conf.get()['model']['se_ratio'] = int(my_conf.get()['model']['se_ratio'])
        # my_conf.get()['model']['lr'] = config['lr']
        # my_conf.get()['optimizer']['name'] = config['optimizer']
        # my_conf.get()['scheduler']['name'] = config['scheduler']
        # my_conf.get()['model']['config'] = np.array(config['network_block_cfg']).reshape(-1,4).tolist()

        my_runner = Runner(my_conf)

        train_once(my_runner)

        my_mean_accuracy =  my_runner.best_acc_top1
        tune.report(mean_accuracy=my_mean_accuracy)

    ray.init(configure_logging=False)

    search_config = {
            #  "conf_dir_file": hp.choice('conf_dir_file', [conf_dir_file]),
             "input_ch": (16, 32),
             "final_ch": (180, 320),
             "width_mult": (1.0, 3.0),
             "depth_mult": (1.0, 3.0),
             "use_se": (False, True),
             "se_ratio": (6, 24), 
             "dropout_ratio": (0.1, 0.5),
            #  "bn_momentum": (0.1, 0.9),
             "lr": (0.001, 0.125)
            #  "optimizer": tune.choice(['SGD','SGDP','Adam','AdamP']),
            #  "scheduler": tune.choice(['CosineAnnealingLR','MultiStepLR']),
            #  "network_block_cfg": tune.grid_search([
            #                                    [2.5, 20, 2, 1,
            #                                     2.5, 36, 1, 2,
            #                                     2.5, 36, 1, 1,
            #                                     2.5, 56, 3, 1,
            #                                     2.5, 80, 1, 2,
            #                                     2.5, 80, 4, 1,
            #                                     2.5, 88, 1, 2,
            #                                     2.5, 96, 2, 1,
            #                                     2.5, 114, 1, 1],
            #                                    [3, 16, 2, 1,
            #                                     3, 32, 1, 2,
            #                                     3, 32, 1, 1,
            #                                     3, 48, 3, 1,
            #                                     3, 72, 1, 2,
            #                                     3, 72, 4, 1,
            #                                     3, 80, 1, 2,
            #                                     3, 88, 2, 1,
            #                                     3, 106, 1, 1]
            #                                   ])
            }

    bo_config = {
        "num_samples": 100,
        "config": {
            'conf_dir_file': conf_dir_file,
        }
    }

    algo = BayesOptSearch(
        search_config,
        max_concurrent=1,
        metric="mean_accuracy",
        mode="max",
        utility_kwargs={
            "kind": "ucb",
            "kappa": 2.5,
            "xi": 0.0
        }
    )
    scheduler = AsyncHyperBandScheduler(metric='mean_accuracy', mode='max')
    analysis = tune.run(train, scheduler=scheduler, search_alg=algo, resources_per_trial={'gpu': 1}, stop={"train_epoch":3}, **bo_config)

    print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

    print('archtecture_search() Done.')

       
def train(runner):

    num_iter = 1 
    if conf.get()['model']['training']['type'] == 'iterative_train':
        num_iter = 4
        runner.train_epochs = runner.train_epochs // num_iter


    for iter_train in range(num_iter):
        runner.reset_average_meters()

        for epoch in range(runner.train_epochs):

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            runner.model.train()

            for i, data in enumerate(tqdm(runner.tr_loader)):
    
                if runner.train_precision == 'fp16':
                    runner.to_half()

                if conf.get()['data']['dali']['avail']:
                    images = data[0]["data"]
                    labels = data[0]["label"].squeeze().long()
                else:
                    images, labels = data
                    if runner.train_precision == 'fp16':
                        images = images.half()

                if conf.get()['cuda']['avail']:
                    images, labels  = images.to(runner.device), labels.to(torch.device(runner.device))
                    runner.model = runner.model.to(torch.device(runner.device))
            
                runner.optimizer.zero_grad()

                outputs, loss = runner.regularizer(images, labels)
            
                loss.backward()
   
                if runner.train_precision == 'fp16':
                    runner.to_half()

                runner.optimizer.step()

                ttop1, ttop5 = runner.accuracy(outputs, labels, (1, 5))

                losses.update(loss.item(), images.size(0))
                top1.update(ttop1.item(), images.size(0))
                top5.update(ttop5.item(), images.size(0))

            print('[{:d}/{:d}] <<<TRAIN>>> lr({:.10f}) loss({:.4f}) top1({:.3f}) top5({:.3f})'.format(
                epoch + 1, runner.train_epochs, runner.optimizer.param_groups[0]['lr'], losses.avg, top1.avg, top5.avg)
            )

            runner.evaluate("train",epoch)
            runner.scheduler.step()
            if conf.get()['data']['dali']['avail']:
                runner.tr_loader.reset()
                runner.test_loader.reset()


        if conf.get()['model']['training']['type'] == 'iterative_train':
            runner.tr_loader, _ , runner.test_loader = get_dataloader(conf)

    print('train() Done.')

def train_prune(runner):
    model_prune_rate, weight_incremental = get_weight_fractions_with_target_ratio(prune_rate=runner.prune_rate, \
                                                                                  pruning_iterations=runner.prune_iterations, \
                                                                                  pruning_target_ratio=runner.prune_target_ratio) 

    print('Pruning iterations :', runner.prune_iterations , ', factions : ', model_prune_rate)

    runner.reset_average_meters()

    for prune in range(runner.prune_iterations):

        if prune == 0:
            # load "train" model and convert it into "train_pruned" model
            model_path =conf.get()['model']['path'] + "/"+ conf.get()['model']['name']
            checkpoint_file =  model_path + "-tr-best.pt"
            print("Load baseline checkpoint:", checkpoint_file)
            checkpoint = torch.load(checkpoint_file)
            runner.model.load_state_dict(checkpoint['model_state_dict'])
            runner.micro_score()

        else:
            checkpoint_file =   conf.get()['model']['path'] + "/"+ conf.get()['model']['name'] + "-pr-" +str(prune-1) +".pt"
            print("Load baseline checkpoint:", checkpoint_file)
            ckpt = runner.load_checkpoint("train_prune", str(prune-1))
            runner.model.load_state_dict(ckpt)
            runner.micro_score()


        for epoch in range(runner.prune_epochs):
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            runner.model.train()

            if epoch == 0:
                masks = weight_prune(runner.model, model_prune_rate[prune])
                runner.model.set_masks(masks)
                print("Prune set_masks() : ", prune , "th iterations")


            for i, data in enumerate(tqdm(runner.tr_loader)):
                images, labels = data
                if conf.get()['cuda']['avail']:
                    images, labels  = images.to(runner.device), labels.to(torch.device(runner.device))
                    runner.model = runner.model.to(torch.device(runner.device))
            
                runner.optimizer.zero_grad()
            
                outputs, loss = runner.regularizer(images, labels)
            
                loss.backward()

                runner.optimizer.step()

                ttop1, ttop5 = runner.accuracy(outputs, labels, (1, 5))

                losses.update(loss.item(), images.size(0))
                top1.update(ttop1.item(), images.size(0))
                top5.update(ttop5.item(), images.size(0))

            print('[{:d}/{:d}] lr({:.10f}) loss({:.4f}) top1({:.3f}) top5({:.3f})'.format(
                epoch + 1, runner.prune_epochs, runner.optimizer.param_groups[0]['lr'], losses.avg, top1.avg, top5.avg)
            )
            runner.evaluate("train_prune",epoch)
            runner.scheduler.step()
        runner.save_checkpoint("train_prune", str(prune))

        if conf.get()['model']['pruning']['type'] == 'iterative_prune':
            runner.tr_loader, _ , runner.test_loader = get_dataloader(conf)

#        runner.model.remove_buffers()  # Remove temporary registered buffers(e.g., MaskedLinear, MaskedConv2d)
    print('train_prune() Done.')

def train_quantize(runner):

    runner.reset_average_meters()

    ckpt = runner.load_checkpoint("train", "best")
    runner.model.load_state_dict(ckpt)

    runner.model.fuse_model()
    runner.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(runner.model, inplace=True)

    for epoch in range(runner.quantize_epochs):

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
    
        #train
        runner.model.train()
        for i, data in enumerate(tqdm(runner.tr_loader)):
            images, labels = data
            if conf.get()['cuda']['avail']:
                images, labels  = images.to(runner.device), labels.to(torch.device(runner.device))
                runner.model = runner.model.to(torch.device(runner.device))

            runner.optimizer.zero_grad()

            outputs, loss = runner.regularizer(images, labels)

            loss.backward()

            runner.optimizer.step()

            ttop1, ttop5 = runner.accuracy(outputs, labels, (1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(ttop1.item(), images.size(0))
            top5.update(ttop5.item(), images.size(0))

        print('[{:d}/{:d}] <<<TRAIN>>> lr({:.10f}) loss({:.4f}) top1({:.3f}) top5({:.3f})'.format( \
                epoch + 1, runner.quantize_epochs, runner.optimizer.param_groups[0]['lr'], losses.avg, top1.avg, top5.avg) )

        if epoch > 3:
            # Freeze quantizer parameters
            runner.model.apply(torch.quantization.disable_observer)
        if epoch > 2:
            # Freeze batch norm mean and variance estimates
            runner.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)


        #evaluate
        quantized_model = torch.quantization.convert(runner.model.eval(), inplace=False)
        quantized_model.eval()

        etop1 = AverageMeter()
        etop5 = AverageMeter()

        with torch.no_grad():
            for i, data in enumerate(tqdm(itertools.islice(runner.test_loader,runner.quantize_evals))):
                if conf.get()['data']['dali']['avail']:
                    timages = data[0]["data"]
                    tlabels = data[0]["label"].squeeze().long()
                else:
                    timages = data[0].type(torch.FloatTensor)
                    tlabels = data[1].type(torch.LongTensor)

                if runner.conf.get()['cuda']['avail']:
                    timages, tlabels  = timages.to(runner.device), tlabels.to(runner.device)

                toutputs = runner.model(timages)

                ttop1, ttop5 = runner.accuracy(toutputs, tlabels, (1, 5))

                etop1.update(ttop1.item(), timages.size(0))
                etop5.update(ttop5.item(), timages.size(0))

            if runner.ebest_acc_top1 <= etop1.avg:
                runner.ebest_acc_top1 = etop1.avg
                if runner.ebest_acc_top5 < etop5.avg:
                    runner.ebest_acc_top5 = etop5.avg
                model_path = runner.conf.get()['model']['path'] + "/"+  runner.conf.get()['model']['name']
                model_path += runner.fn_checkpoint("quantize","best")
                torch.jit.save(torch.jit.script(quantized_model), model_path )

            print('[{:d}/{:d}] <<<TEST>>> top1({:.4f}) top5({:.4f}) best-top1({:.4f}) best-top5({:.4f})'.format(
                  epoch + 1, runner.quantize_epochs, etop1.avg, etop5.avg, runner.ebest_acc_top1, runner.ebest_acc_top5))

    print('train_quantize() Done.')


if __name__ == '__main__':
    # configuration
    args = ConfigArgumentParser(conflict_handler='resolve')
    args.add_argument("--now", type=str, default="Working for AI Grand Challenge(2020)")
    config = args.parse_args()

    # architecture search with hyperopt
    if conf.get()['model']['architecture_search']['type'] != 'none':
        runner = Runner(conf)
        conf_dir_file = os.path.abspath(os.getcwd())  + '/' + config.config
        search(runner, conf_dir_file)

    # training
    if conf.get()['model']['training']['type'] != 'none':
        runner = Runner(conf,run_type="train")
        train(runner)

    # pruning (weight pruning)
    if conf.get()['model']['pruning']['type'] != 'none':
        runner = Runner(conf,run_type="prune")
        train_prune(runner)

    # quantization (qat)
    if conf.get()['model']['quantization']['type'] != 'none':
        runner = Runner(conf,run_type="quantize", device_type='cpu')
        train_quantize(runner)

    print('Done.')
