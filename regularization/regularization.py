import numpy as np
import torch
import copy
import scipy.stats
import math


def get_regularization(conf,images, labels, device):

    if conf.get()['regularization']['name'] == 'cutout':
        lam, images, labels_a, labels_b = cutout(conf, images,labels, device)
    elif conf.get()['regularization']['name'] == 'mixup':
        lam, images, labels_a, labels_b = mixup(conf, images,labels, device)
    elif conf.get()['regularization']['name'] == 'cutmix':
        lam, images, labels_a, labels_b = cutmix(conf, images,labels, device)
    elif conf.get()['regularization']['name'] == 'focusmix':
        lam, images, labels_a, labels_b = focusmix(conf, images,labels, device)
    else:
        raise ValueError(conf.get()['regularization']['name'])

    return lam, images, labels_a, labels_b
    

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mix_image(lam, x1, x2):
    bbx1, bby1, bbx2, bby2 = rand_bbox(x1.size(), lam)
    x1[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]
    
    return x1


def cutout(conf, images, labels, device):
    """
    cutout function 
    https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py

    images: (Batch, C, H, W)
    """
    B,C,H,W = images.size()

    lam = np.random.beta(conf.get()['regularization']['beta'], conf.get()['regularization']['beta'])

    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

     
    #mask[B,C, bbx1: bbx2, bby1: bby2] = 0.
    #mask = torch.from_numpy(mask)
    mask = torch.zeros(B,C,bbx2-bbx1,bby2-bby1)

    images[:, :, bbx1:bbx2, bby1:bby2] = mask
    labels_a = labels

    images = images.to(device)

    return lam, images, labels_a, None

def mixup(conf, images, labels, device):
    """
    mixup function from 'mixup: BEYOND EMPIRICAL RISK MINIMIZATION', 
    https://arxiv.org/pdf/1710.09412.pdf
    """
        
    lam = np.random.beta(conf.get()['regularization']['beta'], conf.get()['regularization']['beta'])

    rand_index = torch.randperm(images.size()[0])
    rand_index= rand_index.to(device)

    images_a = images
    labels_a = labels
    labels_b = labels[rand_index]
    images_b = copy.deepcopy(images)
            
    #images = torch.autograd.Variable(lam * images_a + (1-lam)*images_b[rand_index,:,:,:])
    images = lam * images + ( 1 - lam ) * images[rand_index,:,:,:]
    images = images.to(device)
    
    return lam, images, labels_a, labels_b

def cutmix(conf,images, labels, device):
    """
    cutmix function from 'CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features',
    https://arxiv.org/abs/1905.04899

    images: (Batch, C, H, W)
    """
    #generate mixed sample
    lam = np.random.beta(conf.get()['regularization']['beta'], conf.get()['regularization']['beta'])
    rand_index = torch.randperm(images.size()[0])
    rand_index= rand_index.to(device)

    labels_a = labels
    labels_b = labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

    #adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

    #compute output
    images = torch.autograd.Variable(images, requires_grad=True)
    images = images.to(device)

    return lam, images, labels_a, labels_b


def margin_cut_rand_bbox(conf, images , lam):
    B,C,H,W = images.size()
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # 이미지를 ntiles의 square로 보고, 중심점에서의 exclude_tiles 영역이외의 영역에 집중하도록 함
    # 7등분(7x7)하여 centroid 3개를 exclusive region(3x3)으로 활용
    # 10등분(10x10)하여 centroid 3개를 exclusive region(3x3)으로 활용
    div_ntiles = conf.get()['regularization']['kdmix']['ntiles']
    div_exclude_tiles = conf.get()['regularization']['kdmix']['exclude_tiles']

    exclude_bx1 = (W/div_ntiles) * ((div_ntiles-div_exclude_tiles)/2.0)
    exclude_bx2 = exclude_bx1 + (W/div_ntiles)*div_exclude_tiles
    exclude_by1 = (H/div_ntiles) * ((div_ntiles-div_exclude_tiles)/2.0)
    exclude_by2 = exclude_by1 + (W/div_ntiles)*div_exclude_tiles


    cx = 0 #np.random.randint(np.int(W/div_ntiles))
    cy = 0 # np.random.randint(np.int(H/div_ntiles))

    while True:
        candidate_cx = np.random.randint(W)
        candidate_cy = np.random.randint(H)

        if (candidate_cx < exclude_bx1 or candidate_cx > exclude_bx2) == True \
           and (candidate_cy < exclude_by1 or candidate_cy > exclude_by2) == True :
                cx = candidate_cx
                cy = candidate_cy
                break

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def centroid_bbox(conf, images , bbx1,bby1,bbx2,bby2):
    B,C,H,W = images.size()
    cbbx1 = np.int(W / 2.0) - np.int((bbx2-bbx1) / 2.0)
    cbby1 = np.int(H / 2.0) - np.int((bby2-bby1) / 2.0)
    cbbx2 = cbbx1 + (bbx2-bbx1)
    cbby2 = cbby1 + (bby2-bby1)
    return cbbx1, cbby1, cbbx2, cbby2


N = 10000000
wald_left = None
wald_right = None

def get_wald_interval(mean, scale):
    global wald_left, wald_right, norm_left, norm_right
    if wald_left is None:
        vals = sorted([ np.random.wald(mean, scale)  for _ in range(N) ])
        wald_left, wald_right = vals[0], vals[-1]
    return wald_left, wald_right


def rand_bbox_soft_selection(size, lam):
    """
    images: (Batch, C, H, W)
    """
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    left, right = scipy.stats.norm(0, 1).interval(0.99) # left = 0, right = maximum val

    cx = np.random.normal(0, 1)
    while cx < left or cx > right:
        cx = np.random.normal(0, 1)

    cy = np.random.normal(0, 1)
    while cy < left or cy > right:
        cy = np.random.normal(0, 1)

    cx = int((cx - left) / (right-left) * (W-cut_w)) + cut_w // 2
    cy = int((cy - left) / (right-left) * (H-cut_h)) + cut_h // 2

    bbx1 = cx - cut_w // 2
    bby1 = cy - cut_h // 2
    bbx2 = cx + cut_w // 2
    bby2 = cy + cut_h // 2
    return (bbx1, bby1, bbx2, bby2)

def rand_bbox_soft_exclude(size, lam, mean, scale):
    """
    images: (Batch, C, H, W)
    """
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    xz = np.random.binomial(1, 0.5)
    yz = np.random.binomial(1, 0.5)

    left, right = get_wald_interval(mean, scale)
    cx = np.random.wald(mean, scale)
    while cx < left or cx > right:
        cx = np.random.wald(mean, scale)

    cy = np.random.wald(mean, scale)
    while cy < left or cy > right:
        cy = np.random.wald(mean, scale)

    if xz > 0:
        cx = int((cx - left) / (right - left) * (W - cut_w)) + cut_w // 2
    else:
        cx = int((W - cut_w // 2) - (cx - left) / (right - left) * (W - cut_w))

    if yz > 0:
        cy = int((cy - left) / (right - left) * (H - cut_h)) + cut_h // 2
    else:
        cy = int((H-cut_h // 2) - (cy - left) / (right - left) * (H - cut_h))

    bbx1 = cx - cut_w // 2
    bby1 = cy - cut_h // 2
    bbx2 = cx + cut_w // 2
    bby2 = cy + cut_h // 2
    return (bbx1, bby1, bbx2, bby2)

def focusmix(conf, images, labels, device):
    B,C,H,W = images.size()

    wald_mean = conf.get()['regularization']['focusmix']['wald_mean']
    wald_scale = conf.get()['regularization']['focusmix']['wald_scale']
    lam = np.random.beta(1.0, 1.0)
    lam_min = conf.get()['regularization']['focusmix']['lamda_min']
    while lam < lam_min:
        lam = np.random.beta(1, 1)
    rand_index = torch.randperm(images.size()[0])
    #if conf.get()['cuda']['avail'] == True:
    #    rand_index= rand_index.to(torch.device(conf.get()['cuda']['device']))
    rand_index= rand_index.to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox_soft_selection(images.size(), lam)
    t_bbx1, t_bby1, t_bbx2, t_bby2 = rand_bbox_soft_exclude(images.size(), lam, wald_mean, wald_scale)

    images[:, :, t_bby1:t_bby2, t_bbx1:t_bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]
    labels_a = labels
    labels_b = labels[rand_index]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

    #compute output
    images = torch.autograd.Variable(images, requires_grad=True)
    #if conf.get()['cuda']['avail'] == True:
    #    images = images.to(torch.device(conf.get()['cuda']['device']))
    images = images.to(device)

    return lam, images, labels_a, labels_b

def cut_paste(conf, src, dst, ntile, lam, cams, inv_cams, rand_index, idx):

    def get_bb(ntile, idx_x, idx_y, interval_x, interval_y):
        left = interval_x * idx_x
        right = interval_x * (idx_x + 1)

        bottom = interval_y * idx_y
        top = interval_y * (idx_y+1)
        return left, right, bottom, top

    def rand_bbox_in_cell(left, right, bottom, top, lam, H, W):
        # uniform
        return bbx1, bbx2, bby1, bby2

    W = src.shape[2]
    H = src.shape[1]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    len_h = src.shape[1] // ntile
    len_w = src.shape[2] // ntile

    src_coord = np.random.choice([d for d in range(ntile*ntile)], 1, p=cams[idx])[0]
    src_h = src_coord // ntile
    src_w = src_coord % ntile
    src_left, src_right, src_bottom, src_top = get_bb(ntile, src_w, src_h, len_w, len_h)

    left, right = scipy.stats.norm(0, 1).interval(0.99) # left = 0, right = maximum val

    cx = np.random.normal(0, 1)
    while cx < left or cx > right:
        cx = np.random.normal(0, 1)

    cy = np.random.normal(0, 1)
    while cy < left or cy > right:
        cy = np.random.normal(0, 1)

    cx = int((cx - left) / (right-left) * len_w) + src_left
    cy = int((cy - left) / (right-left) * len_h) + src_bottom

    src_left = np.clip(cx - cut_w // 2, 0, W)
    src_bottom = np.clip(cy - cut_h // 2, 0, H)
    src_right = np.clip(cx + cut_w // 2, 0, W)
    src_top = np.clip(cy + cut_h // 2, 0, H)

    W_ = src_right - src_left
    H_ = src_top - src_bottom

    while True:
        dst_coord = np.random.choice([d for d in range(ntile*ntile)], 1, p=inv_cams[rand_index[idx]])[0]
        dst_h = dst_coord // ntile
        dst_w = dst_coord % ntile
        dst_left, dst_right, dst_bottom, dst_top = get_bb(ntile, dst_w, dst_h, len_w, len_h)

        cx = np.random.randint(dst_right-dst_left) + dst_left
        cy = np.random.randint(dst_top-dst_bottom) + dst_bottom

        dst_left = cx - W_ // 2
        dst_right = cx + W_ // 2
        dst_bottom = cy - H_ // 2
        dst_top = cy + H_ // 2

        if (dst_top - dst_bottom) < (src_top - src_bottom):
            dst_top += 1

        if (dst_right - dst_left) < (src_right - src_left):
            dst_right += 1

        if dst_left >= 0 and dst_right <= W and dst_bottom >= 0 and dst_top <= H:
            break

    dst[:, dst_bottom:dst_top, dst_left:dst_right] = src[:, src_bottom:src_top, src_left:src_right]
    return dst, W_, H_


def compute_score(images, model):

    # Compute CAM-based score (probability)
    def returnCAM(feature_conv, weight_softmax, class_idx):
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for i, idx in enumerate(class_idx):
            cam = weight_softmax[idx].dot(feature_conv[i].reshape((nc, h*w)))
            cam = cam - np.min(cam)
            cam = cam / np.sum(cam)
            output_cam.append(cam)
        return output_cam
    model.eval()

    # Hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    handle = model._modules.get('relu').register_forward_hook(hook_feature)

    # Get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    # Compute scores
    logit = model(images)
    handle.remove()
    h_x = F.softmax(logit, dim=1).data.squeeze()
    if len(h_x.shape) == 1:
        h_x = torch.reshape(h_x, (1, h_x.shape[0]))
    probs, idx = h_x.sort(1, True)
    idx = idx.cpu().numpy()

    CAMs = returnCAM(features_blobs[0], weight_softmax, idx[:,0])
    return CAMs
