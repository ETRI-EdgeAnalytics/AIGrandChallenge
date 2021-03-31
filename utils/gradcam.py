import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import cv2
from torch.autograd import Function

from collections import Sequence
import os.path as osp
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model):
        self.fmaps = {}
        self.grads = {}
        self.model = model
        self.device = next(model.parameters()).device
        for name, module in self.model.named_modules():
            module.register_forward_hook(self.hook(name))
            module.register_backward_hook(self.save_grads(name))

    def hook(self, name):
        def hook_(module, input, output):
            self.fmaps[name] = output

        return hook_

    def save_grads(self, name):
            def backward_hook(module, grad_in, grad_out):
                self.grads[name] = grad_out[0].detach()

            return backward_hook

    def __call__(self, x):
        return self.model(x)
    
    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        if self.logits.dim() > 1:
            one_hot.scatter_(1, ids, 1.0)
        else:
            one_hot.scatter_(0, ids, 1.0)
        return one_hot

    def forward(self, x):
        self.x = x.requires_grad_()
        self.x_shape = x.shape[2:]
        self.logits = self.model(self.x)
        
        if self.logits.dim() > 1:
            self.probs = F.softmax(self.logits, dim=1)
            return self.probs.sort(dim=1, descending=True)
        else:
            self.probs = F.softmax(self.logits, dim=0)
            return self.probs.sort(dim=0, descending=True)
    
    def backward(self, ids):
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmaps, target_layer)
        grads = self._find(self.grads, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(gcam, self.x_shape, mode='bilinear', align_corners=False)

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = []
    with open("label.txt") as lines:
        for line in lines:
            classes.append(line)
    return classes


def preprocess(image_path):
    mean = [0.6510, 0.5797, 0.5601]
    stdv = [0.1826, 0.1747, 0.1738]
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image

def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().detach().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))

# example code
def main():
    import sys
    sys.path.append('/home/ikhee/workspace/Edge_Analytics/system/challenges/GrandChallenge')
    import model.rexnetv1 as rexnet
    import model.efficientnet as efficientnet
    image_path = '/ssd_data/aihub/product_val/HF020001_상품_신발_운동화캐주얼화_운동화_패션운동화/HF021001_0101_0029.JPG'
    #'/ssd_data/aihub/product_val/HF020001_상품_신발_운동화캐주얼화_운동화_패션운동화/HF021001_0103_1090.JPG'

    classes = get_classtable()
    device = torch.device('cuda:2')
    images, raw_images = load_images([image_path])
    images = torch.stack(images).to(device)

    rex_2 = rexnet.ReXNetV1(width_mult=2.0, classes=245)
    state = torch.load('/home/ikhee/workspace/Edge_Analytics/system/challenges/GrandChallenge/checkpoint/rexnetv1_2.0x-tr-pretrained-lr0.00125-batch64-epoch100-best.pt', map_location={'cuda:3': 'cpu'})['model_state_dict']
    rex_2.load_state_dict(state)
    rex_2.to(device)
    rex_2.eval()

    # print(*list(rex_2.named_modules()), sep='\n')
    gradcam = GradCAM(rex_2)
    probs, ids = gradcam.forward(images)
    gradcam.backward(ids[0])
    layer_name = 'features.19'
    regions = gradcam.generate(layer_name)

    save_gradcam(filename='grad_cam_class_{}.png'.format(classes[ids[0]]), gcam=regions[0,0], raw_image=raw_images[0])

if __name__ == '__main__':
    main()