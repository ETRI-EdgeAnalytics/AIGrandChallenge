import torch


import sys
sys.path.append('../')

from config import *
from config import Config as conf

from model import *

from utils.count import count

from model.micronet import MicroNet

from collections import OrderedDict

import os

from data import *
from tqdm import tqdm
from utils.meter import AverageMeter



def bitmask(net):
    num = 0
    for module in net.parameters():
        if module.ndimension() != 1:
            num += module.numel()
    #1-bit per parameter
    return num/32

def count_nonzero(net):
    num = 0
    for module in net.parameters():
        if module.ndimension() != 1:
            num += torch.sum(torch.abs(module.flatten()) != 0.)
    return num.item()

def micro_score(conf, net, precision = 'Freebie'):
    input = torch.randn(1, 3, 32, 32)
    if conf.get()['cuda']['avail']:
        net = net.to(conf.get()['cuda']['device'])
        input = input.to(conf.get()['cuda']['device'])
    addflops, multflops, params = count(net, inputs=(input, ))


    #use fp-16bit
    #if precision == 'Freebie':
    #    multflops = multflops / 2
    #    params = params / 2
    #    multflops = multflops 
    #    params = params 
    #elif precision == 'INT8':
    #    multflops = multflops / 4
    #    params = params / 4


    #add bit-mask
    params += bitmask(net)
    
    score = (params/36500000 + (addflops + multflops)/10490000000)
    #print('Non zero ratio: {}'.format(non_zero_ratio))
    print('<<<Score>>>: {:.4f}, params: {}/{}'.format(score, params, 11689512))
    return score

def grand_challenge_score(conf,net):
    input = torch.randn(1, 3, 32, 32)
    if conf.get()['cuda']['avail']:
        net = net.to(conf.get()['cuda']['device'])
        input = input.to(conf.get()['cuda']['device'])
    addflops, multflops, params = count(net, inputs=(input, ))

    #use fp32
    # we don't have quantization yet

    # add bit-mask
    params += bitmask(net)

    score = params/11689512

    print('<<<Score>>>: {:.4f}, params: {}/{}'.format(score, params, 11689512))
    return score



def _load_checkpoint(conf, model, suffix):
    model_path =conf.get()['model']['path'] + "/"+ conf.get()['model']['name'] + str(suffix) +".pt"

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def quantize(conf, model):
    """
    https://github.com/wjc852456/pytorch-quant

    method: 'linear' # { none : fp32(vanilla) ,  linear, minmax, log, tanh, scale  }
    param_bits: 8 # bit-width for parameters
    bn_bits: 8    # bit-width for running mean and std
    fwd_bits: 8   # bit-width for layer output
    replace_bn: 0 # decide if replace bn layer
    map_bn: 1     # decide if map bn layer to conv layer
    """
    quantization_method = conf.get()['model']['quantization']

    # replace bn with 1x1 conv
    if quantization_method['replace_bn'] == 1:
        quant.replace_bn(model)

    # map bn to conv
    if quantization_method['map_bn'] == 1:
        quant.bn2conv(model)

    # quantize parameters
    if quantization_method['param_bits'] < 32:
        state_dict = model.state_dict()
        state_dict_quant = OrderedDict()
        sf_dict = OrderedDict()
        sf = 0 # dummy
        for k, v in state_dict.items():
            if 'running' in k: # quantize bn layer
                #print("k:{}, v:\n{}".format(k,v))
                if quantization_method['bn_bits'] >=32:
                    print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits = quantization_method['bn_bits']
            else:
                bits = quantization_method['param_bits']

            if quantization_method['method'] == 'linear':
                sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=0.0) #args.overflow_rate)
                # sf stands for float bits
                v_quant  = quant.linear_quantize(v, sf, bits=bits)
                #if 'bias' in k:
                    #print("{}, sf:{}, quantized value:\n{}".format(k,sf, v_quant.sort(dim=0, descending=True)[0]))
            elif quantization_method['method'] == 'log':
                v_quant = quant.log_minmax_quantize(v, bits=bits)
            elif quantization_method['method'] == 'minmax':
                v_quant = quant.min_max_quantize(v, bits=bits)
            else:
                #v_quant = quant.tanh_quantize(v, bits=bits)
                v_quant = v
            state_dict_quant[k] = v_quant
            print("k={0:<35}, bits={1:<5}, sf={2:d>}".format(k,bits,sf))
        model.load_state_dict(state_dict_quant)

    # quantize forward activation
    if quantization_method['fwd_bits'] < 32:
        model = quant.duplicate_model_with_quant(model,
                                                 bits=quantization_method['fwd_bits'],
                                                 overflow_rate=0.0,
                                                 counter=10,
                                                 type=quantization_method['method'])

    return model

def accuracy(output, target, topk=(1,)):
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


def evaluate(conf, model, test_loader):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            timages = data[0].type(torch.FloatTensor)
            tlabels = data[1].type(torch.LongTensor)

            if conf.get()['cuda']['avail']:
                timages, tlabels  = timages.to(conf.get()['cuda']['device']), tlabels.to(conf.get()['cuda']['device'])

            toutputs = model(timages)

            ttop1, ttop5 = accuracy(toutputs, tlabels, (1, 5))
            top1.update(ttop1.item(), timages.size(0))
            top5.update(ttop5.item(), timages.size(0))

        print('<<<TEST>>> top1({:.4f}) top5({:.4f})'.format(top1.avg, top5.avg))


if __name__ == '__main__':
    # configuration
    args = ConfigArgumentParser(conflict_handler='resolve')
    args.add_argument("--now", type=str, default="Working for AI Grand Challenge(2020)")
    config = args.parse_args()
    print(conf.get()['now'])

    # data
    tr_loader, _ , test_loader = get_dataloader(conf)



    # score
    model_path_tr_train_once_ce ='./checkpoint/micronet-tr-best.train_once_ce.pt' # train once
    model_path_tr_train_iter_ce  ='./checkpoint/micronet-tr-best.train_iter_ce.pt'       # train iteratively

    #model_path_pr ='./checkpoint/micronet-pr-best.pt'
    #model_path_pr ='./checkpoint/micronet-pr-final.pt'
    #model_path_qt ='./checkpoint/micronet_quant-qt-best.pt'

    checkpoint_baseline_train_once_ce = torch.load(model_path_tr_train_once_ce)
    checkpoint_baseline_train_iter_ce = torch.load(model_path_tr_train_iter_ce)
    #checkpoint_pr = torch.load(model_path_pr)
#    checkpoint_qt = torch.load(model_path_qt)

    print("============ <Baseline model score> ================")
    net1 = MicroNet(ver = 'ver1', num_classes = 100, add_se = True, Activation = 'HSwish')
    net1.load_state_dict(checkpoint_baseline_train_once_ce['model_state_dict'], strict=False)
    grand_challenge_score(conf, net1)
    evaluate(conf,net1,test_loader)

    net2 = MicroNet(ver = 'ver1', num_classes = 100, add_se = True, Activation = 'HSwish')
    net2.load_state_dict(checkpoint_baseline_train_iter_ce['model_state_dict'], strict=False)
    grand_challenge_score(conf, net2)
    evaluate(conf,net2,test_loader)

#    print("============ <Pruned model score> ================")
#    net2 = MicroNet(ver = 'ver1', num_classes = 100, add_se = True, Activation = 'HSwish')
#    net2.load_state_dict(checkpoint_pr['model_state_dict'], strict=False)
#    micro_score(conf, net2)
#    evaluate(conf,net2,test_loader)
#
#    model_path =conf.get()['model']['path']
#    pr_files_idx = [ f.replace('micronet-pr-','').replace('.pt','') for f in os.listdir(model_path) if f.startswith('micronet-pr-') ]
#    pr_files_idx = [ x for x in pr_files_idx if x != 'best']
#    pr_files_idx = [ x for x in pr_files_idx if x != 'final']
#    pr_files_idx = [ int(x) for x in pr_files_idx ]
#    pr_files_idx.sort()
#    net3 = MicroNet(ver = 'ver1', num_classes = 100, add_se = True, Activation = 'HSwish')
#    for pr_idx in pr_files_idx:
#        model_path_pr_idx = './checkpoint/micronet-pr-'+ str(pr_idx) + '.pt'
#        checkpoint_pr_idx  = torch.load(model_path_pr_idx)
#        net3.load_state_dict(checkpoint_pr_idx['model_state_dict'], strict=False)
#        micro_score(conf, net3)
#        evaluate(conf,net3,test_loader)

    #net2.load_state_dict(checkpoint_pr['model_state_dict'], strict=False)
    #micro_score(conf, net2)
#
#    net3 = MicroNet(ver = 'ver1', num_classes = 100, add_se = True, Activation = 'HSwish')
#    net3.load_state_dict(checkpoint_pr0['model_state_dict'], strict=False)
#    micro_score(conf, net3)
#
#    net4 = MicroNet(ver = 'ver1', num_classes = 100, add_se = True, Activation = 'HSwish')
#    net4.load_state_dict(checkpoint_pr1['model_state_dict'], strict=False)
#    micro_score(conf, net4)
#
#    net5 = MicroNet(ver = 'ver1', num_classes = 100, add_se = True, Activation = 'HSwish')
#    net5.load_state_dict(checkpoint_pr2['model_state_dict'], strict=False)
#    micro_score(conf, net5)

#    print("============ <Quantized model score> ================")
#    net4 = MicroNet_quant(ver = 'ver1', num_classes = 100, add_se = True, Activation = 'HSwish')
#    net4.load_state_dict(checkpoint_baseline, strict = False)
#    micro_score(conf, net4)
#
#    net5 = MicroNet_quant(ver = 'ver1', num_classes = 100, add_se = True, Activation = 'HSwish')
#    net5.load_state_dict(checkpoint_pr, strict = False)
#    micro_score(conf, net5)
#
#    net6 = MicroNet_quant(ver = 'ver1', num_classes = 100, add_se = True, Activation = 'HSwish')
#    net6.load_state_dict(checkpoint_qt, strict = False)
#    micro_score(conf, net6, "INT8")


    #conf.get()['model']['quantization']['method'] = 'linear'
    #net7 = quantize(conf, net3)
    #micro_score(conf,net7)

