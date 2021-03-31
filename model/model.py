import torchvision.models as models

import torch
import torch.nn as nn

import os


def get_teacher_model(conf):
    if  conf.get()['regularization']['kdmix']['teacher_model'] == "resnet50":
        from .resnet import resnet50
        teacher_model = resnet50()
    elif  conf.get()['regularization']['kdmix']['teacher_model'] == "resnet101":
        from .resnet import resnet101
        teacher_model = resnet101()
    elif  conf.get()['regularization']['kdmix']['teacher_model'] == "resnet152":
        from .resnet import resnet152
        teacher_model = resnet152()
    elif  conf.get()['regularization']['kdmix']['teacher_model'] == "wideresnet":
        from .wideresnet import wideresnet
        teacher_model = wideresnet()
    elif  conf.get()['regularization']['kdmix']['teacher_model'] == "pyramidnet":
        from .pyramidnet import pyramidnet 
        teacher_model = pyramidnet()
    else:
        raise ValueError(conf.get()['regularization']['kdmix']['teacher_model'])
        

    teacher_model_path = conf.get()['regularization']['kdmix']['path']

    if not os.path.exists(teacher_model_path):
        raise("File doesn't exist {}".format(teacher_model_path))
    if conf.get()['cuda']['avail'] == True:
        checkpoint = torch.load(teacher_model_path)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    teacher_model.load_state_dict(checkpoint["model"])

    optimizer = None
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])


    if conf.get()['cuda']['avail'] == True:
        if conf.get()['cuda']['device'] == 'cuda':
            teacher_model = nn.DataParallel(teacher_model)
            teacher_model = teacher_model.to(torch.device('cuda'))
        else:
            teacher_model = teacher_model.to(torch.device(conf.get()['cuda']['device']))

    return teacher_model



def get_model(conf, ver_type=None, run_type= None, device_type=None):
    if conf.get()['model']['name'] == 'vgg16':
        from .vgg import vgg16_bn
        model = vgg16_bn()
    elif conf.get()['model']['name'] == 'vgg13':
        from .vgg import vgg13_bn
        model = vgg13_bn()
    elif conf.get()['model']['name'] == 'vgg11':
        from .vgg import vgg11_bn
        model = vgg11_bn()
    elif conf.get()['model']['name'] == 'vgg19':
        from .vgg import vgg19_bn
        model = vgg19_bn()
    elif conf.get()['model']['name'] == 'densnet121':
        from .densenet import densenet121 
        model = densenet121()
    elif conf.get()['model']['name'] == 'densnet161':
        from .densenet import densenet161 
        model = densenet161()
    elif conf.get()['model']['name'] == 'densnet169':
        from .densenet import densenet169 
        model = densenet169()
    elif conf.get()['model']['name'] == 'densnet201':
        from .densenet import densenet201
        model = densenet201()
    elif conf.get()['model']['name'] == 'googlenet':
        from .googlenet import googlenet 
        model = googlenet()
    elif conf.get()['model']['name'] == 'inceptionv3':
        from .inceptionv3 import inceptionv3 
        model = inceptionv3()
    elif conf.get()['model']['name'] == 'inceptionv4':
        from .inceptionv4 import inceptionv4 
        model = inceptionv4()
    elif conf.get()['model']['name'] == 'inception_resnet_v2':
        from .inceptionv4 import inception_resnet_v2
        model = inception_resnet_v2()
    elif conf.get()['model']['name'] == 'xception':
        from .xception import xception 
        model = xception()
    elif conf.get()['model']['name'] == 'resnet18':
        from .resnet import resnet18
        model = resnet18()
    elif conf.get()['model']['name'] == 'resnet34':
        from .resnet import resnet34
        model = resnet34()
    elif conf.get()['model']['name'] == 'resnet50':
        from .resnet import resnet50
        model = resnet50()
    elif conf.get()['model']['name'] == 'resnet101':
        from .resnet import resnet101
        model = resnet101()
    elif conf.get()['model']['name'] == 'resnet152':
        from .resnet import resnet152
        model = resnet152()
    elif conf.get()['model']['name'] == 'resnext50':
        from .resnext import resnext50
        model = resnext50()
    elif conf.get()['model']['name'] == 'resnext101':
        from .resnext import resnext101
        model = resnext101()
    elif conf.get()['model']['name'] == 'resnext152':
        from .resnext import resnext152
        model = resnext152()
    elif conf.get()['model']['name'] == 'shufflenet':
        from .shufflenet import shufflenet
        model = shufflenet()
    elif conf.get()['model']['name'] == 'shufflenetv2':
        from .shufflenetv2 import shufflenetv2
        model = shufflenetv2()
    elif conf.get()['model']['name'] == 'squeezenet':
        from .squeezenet import squeezenet 
        model = squeezenet()
    elif conf.get()['model']['name'] == 'mobilenet':
        from .mobilenet import mobilenet 
        model = mobilenet()
    elif conf.get()['model']['name'] == 'mobilenetv2':
        from .mobilenetv2 import mobilenetv2 
        model = mobilenetv2()
    elif conf.get()['model']['name'] == 'nasnet':
        from .nasnet import nasnet 
        model = nasnet()
    elif conf.get()['model']['name'] == 'wideresnet':
        from .wideresnet import wide_resnet50_2
        model = wide_resnet50_2()
    elif conf.get()['model']['name'] == 'pyramidnet':
        from .pyramidnet import pyramidnet 
        model = pyramidnet()
    elif conf.get()['model']['name'] == 'micronet':
        if run_type == "quantize":
            from .micronet_quant import micronet
        else:
            if '-cbam' in conf.get()['model']['ver']:
                from .micronet_cbam import micronet
            else:
                from .micronet import micronet
        if conf.get()['model']['ver'] == None:
            model = micronet(conf,ver="ver1", run_type=run_type, device_type=device_type)
        else:
            model = micronet(conf,ver=ver_type, run_type=run_type,  device_type=device_type)
    elif conf.get()['model']['name'] == 'mixnet':
        from .mixnet import mixnet_s
        model = mixnet(conf)

    # 
    #Custom pretrained model architectures for image classification
    #
    elif conf.get()['model']['name'].startswith('efficientnet-b') == True:
        from .efficientnet import EfficientNet
        if conf.get()['model']['name'] == 'efficientnet-b0':   # 5.3M, 76.3%(top1)
            model = EfficientNet.from_pretrained('efficientnet-b0',weights_path=conf.get()['model']['path'])
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(512*1, conf.get()['model']['num_class'])
        elif conf.get()['model']['name'] == 'efficientnet-b1': # 7.8M, 78.8%(top1)
            model = EfficientNet.from_pretrained('efficientnet-b1',weights_path=conf.get()['model']['path'])
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(512*1, conf.get()['model']['num_class'])
        elif conf.get()['model']['name'] == 'efficientnet-b2': # 9.2M, 79.8%(top1)
            model = EfficientNet.from_pretrained('efficientnet-b2',weights_path=conf.get()['model']['path'])
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(512*1, conf.get()['model']['num_class'])
        elif conf.get()['model']['name'] == 'efficientnet-b3': #  12M, 81.1%(top1)
            model = EfficientNet.from_pretrained('efficientnet-b3',weights_path=conf.get()['model']['path'])
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(512*1, conf.get()['model']['num_class'])
        elif conf.get()['model']['name'] == 'efficientnet-b4': #  19M, 82.6%(top1)
            model = EfficientNet.from_pretrained('efficientnet-b4',weights_path=conf.get()['model']['path'])
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(512*1, conf.get()['model']['num_class'])
        elif conf.get()['model']['name'] == 'efficientnet-b5': #  30M, 83.3%(top1)
            model = EfficientNet.from_pretrained('efficientnet-b5',weights_path=conf.get()['model']['path'])
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(512*1, conf.get()['model']['num_class'])
        elif conf.get()['model']['name'] == 'efficientnet-b6': #  43M, 84.0%(top1)
            model = EfficientNet.from_pretrained('efficientnet-b6',weights_path=conf.get()['model']['path'])
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(512*1, conf.get()['model']['num_class'])
        elif conf.get()['model']['name'] == 'efficientnet-b7': #  66M, 84.4%(top1)
            model = EfficientNet.from_pretrained('efficientnet-b7',weights_path=conf.get()['model']['path'])
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(512*1, conf.get()['model']['num_class'])
        else:
            raise ValueError(conf.get()['model']['name'])
            
    elif conf.get()['model']['name'].startswith('rexnetv1_') == True and "diff" not in conf.get()['model']['name']:
        if run_type == "quantize":
            from .rexnetv1_quant import ReXNetV1
        else:
            from .rexnetv1 import ReXNetV1
        if conf.get()['model']['name'] == 'rexnetv1_0.9x':  # 4.8M 77.9%(top1)
            model = ReXNetV1(width_mult=0.9, classes=conf.get()['model']['num_class'])
        elif conf.get()['model']['name'] == 'rexnetv1_1.0x':  # 4.8M 77.9%(top1)
            model = ReXNetV1(width_mult=1.0, classes=conf.get()['model']['num_class'])
            if conf.get()['model']['pretrained']:
                path = conf.get()['model']['path'].replace("checkpoint", "model/pretrained")
                path = os.path.join(path, "rexnetv1_1.0x.pth")
                dict_ = torch.load(path, map_location=torch.device('cpu'))
                dict_.pop("output.1.weight")
                dict_.pop("output.1.bias")
                model_dict = model.state_dict()
                for key in dict_: # override
                    model_dict[key] = dict_[key]
                model.load_state_dict(model_dict)

        elif conf.get()['model']['name'] == 'rexnetv1_1.3x':  # 7.6M 79.5%(top1)
            model = ReXNetV1(width_mult=1.3, classes=conf.get()['model']['num_class'])
            if conf.get()['model']['pretrained']:
                path = conf.get()['model']['path'].replace("checkpoint", "model/pretrained")
                path = os.path.join(path, "rexnetv1_1.3x.pth")
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        elif conf.get()['model']['name'] == 'rexnetv1_1.5x':  # 9.7M 80.3%(top1)
            model = ReXNetV1(width_mult=1.5, classes=conf.get()['model']['num_class'])
            if conf.get()['model']['pretrained']:
                path = conf.get()['model']['path'].replace("checkpoint", "model/pretrained")
                path = os.path.join(path, "rexnetv1_1.5x.pth")
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        elif conf.get()['model']['name'] == 'rexnetv1_2.0x':  # 19M 81.7%(top1)
            model = ReXNetV1(width_mult=2.0, classes=1000)
            if conf.get()['model']['pretrained']:
                path = conf.get()['model']['path'].replace("checkpoint", "model/pretrained")
                path = os.path.join(path, "rexnetv1_2.0x.pth")
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                model.output = nn.Sequential(
                                    nn.Dropout(0.2),
                                    nn.Conv2d(2560, conf.get()['model']['num_class'], 1, bias=True))
        elif conf.get()['model']['name'] == 'rexnetv1_search':
            model = ReXNetV1(width_mult=conf.get()['model']['width_mult'], depth_mult=conf.get()['model']['depth_mult'], classes=conf.get()['model']['num_class'],
                use_se=conf.get()['model']['use_se'], se_ratio=conf.get()['model']['se_ratio'], dropout_ratio=conf.get()['model']['dropout_ratio']
            )
        else:
            raise ValueError(conf.get()['model']['name'])

    elif conf.get()['model']['name'].startswith('rexnetv1CBAM') == True:
        from .rexnetv1_cbam import ReXNetV1CBAM
        model = ReXNetV1CBAM(width_mult=1.0, classes=conf.get()['model']['num_class'])
        if conf.get()['model']['pretrained']:
            path = conf.get()['model']['path'].replace("checkpoint", "model/pretrained")
            path = os.path.join(path, "rexnetv1_1.0x.pth")
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        elif conf.get()['model']["weight_path"] != "none":
            path = conf.get()['model']["weight_path"]
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    elif conf.get()['model']['name'].startswith('rexnetv1_diff') == True:
        from .rexnetv1_diff import ReXNetV1
        model = ReXNetV1(width_mult=1.0, classes=conf.get()['model']['num_class'])
        if conf.get()['model']['pretrained']:
            path = conf.get()['model']['path'].replace("checkpoint", "model/pretrained")
            path = os.path.join(path, "rexnetv1_1.0x.pth")
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    else:

        raise ValueError(conf.get()['model']['name'])

    if conf.get()['cuda']['avail'] == True:
        if conf.get()['cuda']['device'] == 'cuda':
            model = nn.DataParallel(model)
            model = model.to(torch.device('cuda'))
        else:
            device = torch.device(conf.get()['cuda']['device'])
            model = model.to(device)

    return model


