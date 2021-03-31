from torch.optim import Adam, SGD

from .adamp import AdamP
from .sgdp import SGDP
from .ranger import Ranger


def get_optimizer(model, conf):
    #optimizer_amsgrad=False, optimizer_betas='(0.9, 0.999)', optimizer_eps='1e-08', optimizer_name='Adam', optimizer_weight_decay=0

    if "flr" not in conf.get()['model'] or conf.get()['model']['flr'] == conf.get()['model']['lr']:
        parameters = model.parameters()
    else:
        parameters = [{"params":model.features.parameters(), "lr":conf.get()['model']['flr']}, {"params":model.output.parameters()}]

    if conf.get()['optimizer']['name'] == 'Adam':
        optimizer = Adam(
            parameters, 
            lr=conf.get()['model']['lr'], 
            weight_decay=conf.get()['optimizer']['weight_decay'])

    elif conf.get()['optimizer']['name'] == 'SGD':
        optimizer = SGD(
            parameters,
            lr=conf.get()['model']['lr'],
            momentum=conf.get()['optimizer']['momentum'],
            weight_decay=conf.get()['optimizer']['weight_decay'],
            nesterov=(conf.get()['optimizer']['weight_decay'] > 0)
        )
    # NAVER Corp. from https://github.com/clovaai/adamp
    elif conf.get()['optimizer']['name'] == 'AdamP':
        optimizer = AdamP(
            parameters,
            lr=conf.get()['model']['lr'], 
            betas=(0.9, 0.999), 
            weight_decay=conf.get()['optimizer']['weight_decay']
         ) 
    # NAVER Corp. from https://github.com/clovaai/adamp
    elif conf.get()['optimizer']['name'] == 'SGDP':
        optimizer = SGDP(
            parameters,
            lr=conf.get()['model']['lr'], 
            weight_decay=conf.get()['optimizer']['weight_decay'],
            momentum=conf.get()['optimizer']['momentum'],
            nesterov=(conf.get()['optimizer']['weight_decay'] > 0)
         ) 
    # https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
    elif conf.get()['optimizer']['name'] == 'Ranger':
        optimizer = Ranger(
            parameters,
            lr=conf.get()['model']['lr']
         ) 
    else:
        raise ValueError(conf.get()['optimizer']['name'])
    return optimizer
