import os
import numpy as np
import pickle

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

import sys
sys.path.append('../')
from model.rexnetv1 import ReXNetV1

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


model = ReXNetV1(width_mult=2.0 ,  classes=245)
ckpt = torch.load('checkpoint_ikhee/rexnetv1_2.0x-tr-pretrained-lr0.00125-batch64-epoch100-best.pt' , map_location='cuda:0')
model.load_state_dict(ckpt['model_state_dict'])

print(model)
print('---original---')
print_size_of_model(model)

'''
ReXNetV1(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Swish(
      (sigmoid): Sigmoid()
    )
    (3): LinearBottleneck(
      (out): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6()
        (3): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )

    ...
'''

#for name, module in model.named_modules():
#    # prune 20% of connections in all 2D-conv layers
#    if isinstance(module, torch.nn.Conv2d):
#        prune.l1_unstructured(module, name='weight', amount=0.2)
#        prune.remove(module,'weight')
#    # prune 40% of connections in all linear layers
#    elif isinstance(module, torch.nn.Linear):
#        prune.l1_unstructured(module, name='weight', amount=0.2)
#        prune.remove(module,'weight')
#
#print('---pruned/0.2 ---')
#print_size_of_model(model)

def save_object(data,fn):
   with open(fn, 'wb') as f:
       pickle.dump(data, f)

def load_object(data,fn):
   with open(fn, 'rb') as f:
       pickle.load(data, f)

pr = 92.0
pruning_perc = pr
all_weights = []
for n, p in model.named_parameters():
    if len(p.data.size()) != 1:
        print(n)
        all_weights += list(p.cpu().data.abs().numpy().flatten())
threshold = np.percentile(np.array(all_weights),pruning_perc)
print('threshold:' + str(threshold))

masks = []
number_of_pruned = []
for p in model.parameters():
    if len(p.data.size()) != 1:
        pruned_inds = p.data.abs() > threshold
        masks.append(pruned_inds.float())

total = len(all_weights)
elimination = []
for i in np.array(all_weights):
    if i < threshold:
        elimination.append(i)
cut = len(elimination)
print('# of cut:' + str(cut)+'/'+str(total), ', # of remaining:' + str(total-cut))

all_weights = []
for p in model.parameters():
    if len(p.data.size()) != 1:
        all_weights += list(p.cpu().data.numpy())
save_object(all_weights,'./checkpoint_ikhee/ckpt-pruned-dense-'+str(pr)+'.pkl')
        
#pruning
model.set_masks(masks)

sparse_all_weights = []
for p in model.parameters():
    if len(p.data.size()) != 1:
        shapes = np.asarray(p.data.shape)
        data_flatten = p.cpu().data.flatten().numpy()
        indices = np.nonzero(data_flatten)
        values = np.array(tuple(data_flatten[i] for i in indices))
        sparse_all_weights += list( [shapes, indices, values] )
    else:
        data_flatten = p.cpu().data.flatten().numpy()
        sparse_all_weights += list([data_flatten])

save_object(sparse_all_weights,'./checkpoint_ikhee/ckpt-pruned-sparse-'+str(pr)+'.pkl')
