import os
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms


from .autoaugment import CIFAR10Policy, ImageNetPolicy
from .fast_augmentations import *
from .cifar100_fast_aug import fa_shake26_2x96d_cifar100, fa_wresnet40x2_cifar100_r5, fa_wresnet40x2_cifar100, fa_wresnet28x10_cifar100

from .imbalanced import ImbalancedDatasetSampler

####### CIFAR-100
# https://github.com/clovaai/overhaul-distillation/blob/master/CIFAR-100/train_with_distillation.py
def get_dataloader_cifar100(conf):

    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762))
    ])


    trainset = datasets.CIFAR100(conf.get()['data']['tr']['path'], train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(
        trainset , batch_size=conf.get()['model']['batch'], shuffle=True, num_workers=4)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762))
    ])

    testset = datasets.CIFAR100(conf.get()['data']['test']['path'], train=False, download=True, transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=conf.get()['model']['batch'], shuffle=False, num_workers=4)

    return train_loader, None , test_loader


def get_dataloader_cifar100_autoaugment(conf):

    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762))
    ])
    trainset = datasets.CIFAR100(conf.get()['data']['tr']['path'], train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset , batch_size=conf.get()['model']['batch'], shuffle=True, num_workers=4)


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762))
    ])
    testset = datasets.CIFAR100(conf.get()['data']['test']['path'], train=False, download=True, transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=conf.get()['model']['batch'], shuffle=False, num_workers=4)

    return train_loader, None , test_loader


def get_dataloader_cifar100_fast_autoaugment(conf):
    transform_train1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill = 128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762))
        ])
    transform_train2 = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill = 128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762))
        ])
    transform_train3 = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill = 128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762))
        ])
    transform_train4 = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill = 128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762))
        ])

    transform_train1.transforms.insert(0, Augmentation(fa_wresnet28x10_cifar100()))
    transform_train2.transforms.insert(0, Augmentation(fa_shake26_2x96d_cifar100()))
    transform_train3.transforms.insert(0, Augmentation(fa_wresnet40x2_cifar100()))
    transform_train4.transforms.insert(0, Augmentation(fa_wresnet40x2_cifar100_r5()))

    trainset = datasets.CIFAR100(conf.get()['data']['tr']['path'], train=True, download=True, transform=transform_train1)
    trainset += datasets.CIFAR100(conf.get()['data']['tr']['path'], train=True, download=False, transform=transform_train2)
    trainset += datasets.CIFAR100(conf.get()['data']['tr']['path'], train=True, download=False, transform=transform_train3)
    trainset += datasets.CIFAR100(conf.get()['data']['tr']['path'], train=True, download=False, transform=transform_train4)

    train_loader = data.DataLoader(trainset , batch_size=conf.get()['model']['batch'], shuffle=True, num_workers=4)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762))
    ])
    testset = datasets.CIFAR100(conf.get()['data']['test']['path'], train=False, download=True, transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=conf.get()['model']['batch'], shuffle=False, num_workers=4)

    return train_loader, None , test_loader

def get_dataloader_imagenet(conf):
    """
    Only train and valid set exists for imagenet dataset.
    train/test transfromations are following conventions.
    Imagenet Dataset has same images for Classification Task (2012-2017)
    The valid images are from ImageNet2012. 
    """
    if conf.get()['data']['dali']['avail']:
        return get_dali_imagenet(conf)

    mean = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv)
    ])

    trainset = datasets.ImageFolder(conf.get()['data']['tr']['path'], transform=transform_train)
    train_loader = data.DataLoader(
        trainset , batch_size=conf.get()['model']['batch'], shuffle=True, num_workers=4)

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv)
    ])

    testset = datasets.ImageFolder(conf.get()['data']['test']['path'], transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=conf.get()['model']['batch'], shuffle=False, num_workers=4)

    return train_loader, None , test_loader

def get_dataloader_product(conf):
    """

    """
    if conf.get()['data']['dali']['avail']:
        return get_dali_imagenet(conf)

    mean = [0.6510, 0.5797, 0.5601]
    stdv = [0.1826, 0.1747, 0.1738]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv)
    ])

    trainset = datasets.ImageFolder(conf.get()['data']['tr']['path'], transform=transform_train)
    train_loader = data.DataLoader(
        # yjlee: sampler don't have any shuffle operation.
        # trainset, batch_size=conf.get()['model']['batch'], shuffle=True, num_workers=4)
        trainset, sampler=ImbalancedDatasetSampler(trainset),batch_size=conf.get()['model']['batch'], num_workers=4)

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv)
    ])

    testset = datasets.ImageFolder(conf.get()['data']['test']['path'], transform=transform_test)
    # yjlee: sampler don't have any shuffle operation.
    #test_loader = data.DataLoader(testset, batch_size=conf.get()['model']['batch'], shuffle=False, num_workers=4)
    test_loader = data.DataLoader(testset, sampler=ImbalancedDatasetSampler(testset),batch_size=conf.get()['model']['batch'], num_workers=4)

    return train_loader, None , test_loader

def get_dataloader_product_autoaugment(conf):
    """

    """
    if conf.get()['data']['dali']['avail']:
        return get_dali_imagenet(conf)

    mean = [0.6510, 0.5797, 0.5601]
    stdv = [0.1826, 0.1747, 0.1738]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv)
    ])

    trainset = datasets.ImageFolder(conf.get()['data']['tr']['path'], transform=transform_train)
    train_loader = data.DataLoader(
        # yjlee: sampler don't have any shuffle operation.
        # trainset, batch_size=conf.get()['model']['batch'], shuffle=True, num_workers=4)
        trainset, sampler=ImbalancedDatasetSampler(trainset),batch_size=conf.get()['model']['batch'], num_workers=4)

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv)
    ])

    testset = datasets.ImageFolder(conf.get()['data']['test']['path'], transform=transform_test)
    # yjlee: sampler don't have any shuffle operation.
    #test_loader = data.DataLoader(testset, batch_size=conf.get()['model']['batch'], shuffle=False, num_workers=4)
    test_loader = data.DataLoader(testset, sampler=ImbalancedDatasetSampler(testset),batch_size=conf.get()['model']['batch'], num_workers=4)

    return train_loader, None , test_loader

def get_dataloader_product_randaugment(conf):
    """

    """
    from .randaugment import RandAugment
    # if conf.get()['data']['dali']['avail']:
    #     return get_dali_imagenet(conf)

    mean = [0.6510, 0.5797, 0.5601]
    stdv = [0.1826, 0.1747, 0.1738]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv)
    ])

    transform_train.transforms.insert(0, RandAugment(conf.get()['data']['augmentation']['N'], conf.get()['data']['augmentation']['M']))
    trainset = datasets.ImageFolder(conf.get()['data']['tr']['path'], transform=transform_train)
    train_loader = data.DataLoader(
        #trainset , batch_size=conf.get()['model']['batch'], shuffle=True, num_workers=4)
        trainset, sampler=ImbalancedDatasetSampler(trainset),batch_size=conf.get()['model']['batch'], num_workers=4)

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv)
    ])

    testset = datasets.ImageFolder(conf.get()['data']['test']['path'], transform=transform_test)
    #test_loader = data.DataLoader(testset, batch_size=conf.get()['model']['batch'], shuffle=False, num_workers=4)
    test_loader = data.DataLoader(testset, sampler=ImbalancedDatasetSampler(testset), batch_size=conf.get()['model']['batch'], shuffle=False, num_workers=4)

    return train_loader, None , test_loader

def get_dali_imagenet(conf):
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from .dali import HybridTrainPipe, HybridValPipe

    dali_cpu = False if conf.get()['cuda']['avail'] else True
    pipe = HybridTrainPipe(batch_size=conf.get()['model']['batch'], num_threads=4, device_id=0, data_dir=conf.get()['data']['tr']['path'], crop=224, dali_cpu=dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))

    pipe = HybridValPipe(batch_size=conf.get()['model']['batch'], num_threads=4, device_id=0, data_dir=conf.get()['data']['test']['path'], crop=224, size=256)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))

    return train_loader, None, val_loader

def get_dataloader_imagenet_autoaugment(conf):
    transform_train = transforms.Compose([
        ImageNetPolicy(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225))
    ])

    trainset = datasets.ImageFolder(conf.get()['data']['tr']['path'], transform=transform_train)
    train_loader = data.DataLoader(
        trainset , batch_size=conf.get()['model']['batch'], shuffle=True, num_workers=4)

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225))
    ])

    testset = datasets.ImageFolder(conf.get()['data']['test']['path'], transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=conf.get()['model']['batch'], shuffle=False, num_workers=4)

    return train_loader, None , test_loader

def get_dataloader(conf):
    if conf.get()['data']['type'] == 'CIFAR-100':
        if conf.get()['data']['augmentation']['name'] == 'none':
            return get_dataloader_cifar100(conf)
        elif conf.get()['data']['augmentation']['name'] == 'autoaugment':
            return get_dataloader_cifar100_autoaugment(conf)
        elif conf.get()['data']['augmentation']['name'] == 'fast-autoaugment':
            return get_dataloader_cifar100_fast_autoaugment(conf)
        else:
            raise ValueError(conf.get()['data']['augmentation']['name'] + " is not yet !!!")
    elif conf.get()['data']['type'] == 'ImageNet':
        if conf.get()['data']['augmentation']['name'] == 'none':
            return get_dataloader_imagenet(conf)
        elif conf.get()['data']['augmentation']['name'] == 'autoaugment':
            return get_dataloader_imagenet_autoaugment(conf)
        # elif conf.get()['data']['augmentation']['name'] == 'fast=-autoaugment:
        #     return get_dataloader_imagenet_fast_autoaugment(conf)
        else:
            raise ValueError(conf.get()['data']['augmentation']['name'] + " is not yet !!!")
    elif conf.get()['data']['type'] == 'GrandChallenge':
        if conf.get()['data']['augmentation']['name'] == 'none':
            return get_dataloader_product(conf)
        elif conf.get()['data']['augmentation']['name'] == 'autoaugment':
            return get_dataloader_product_autoaugment(conf)
        elif conf.get()['data']['augmentation']['name'] == 'randaugment':
            return get_dataloader_product_randaugment(conf)
        # elif conf.get()['data']['augmentation']['name'] == 'fast=-autoaugment:
        #     return get_dataloader_imagenet_fast_autoaugment(conf)
        else:
            raise ValueError(conf.get()['data']['augmentation']['name'] + " is not yet !!!")
    else:
        raise ValueError(conf.get()['data']['type'] + " is not yet !!!")

