import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import copy


def get_prune_percent(prune,total):
    """
    Argument
    --------
    total : The number whose 20 percent we need to calculate
    Returns
    -------
    20% of total
    """
    return prune*0.01*total


def get_weight_fractions(prune_rate, pruning_iterations):
    """
    Returns a list of numbers which represent the fraction of weights pruned after each pruning iteration
    """
    percent_20s = []
    for i in range(1, pruning_iterations+1):
        percent_20s.append(get_prune_percent(prune_rate,(100- sum(percent_20s))))
    weight_fractions = []
    for i in range(1,pruning_iterations+1):
        weight_fractions.append(sum(percent_20s[:i]))
    return weight_fractions



def get_weight_fractions_with_target_ratio(prune_rate, pruning_iterations,pruning_target_ratio):
    """
    Returns a list of numbers which represent the fraction of weights pruned after each pruning iteration
    """
    percent_20s = []
    #target_pruning_ratio = 70 #100
    for i in range(1, pruning_iterations+1):
        percent_20s.append(get_prune_percent(prune_rate,(pruning_target_ratio - sum(percent_20s))))
    weight_fractions = []
    weight_incremental = []
    for i in range(1,pruning_iterations+1):
        weight_fractions.append(sum(percent_20s[:i]))
        weight_incremental.append(percent_20s[i-1])
    return weight_fractions,weight_incremental


#model_prune_iterations = 5
#model_prune_rate = [ 15., 30., 45. , 60., 75. ]
#weight_fractions = get_weight_fractions(prune_rate=20,pruning_iterations=10)
#print(weight_fractions)
#weight_fractions,weight_incremental = get_weight_fractions_with_target_ratio(prune_rate=25,pruning_iterations=15, pruning_target_ratio=70)
#print(weight_fractions)
#print(weight_fractions,weight_incremental)



def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(torch.abs(p).cpu().data.numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = torch.abs(p).data > threshold
            masks.append(pruned_inds.float())
    return masks



def prune_by_percentile(model, mask, percent, resample=False, reinit=False):
    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            weight_dev = param.device
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    step = 0
    return model, mask

def make_mask(model):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None]* step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0
    return mask


def permute_masks(old_masks):
    """
    Function to randomly permute the mask in a global manner.
    Arguments
    ---------
    old_masks: List containing all the layer wise mask of the neural network, mandatory. No default.
    seed: Integer containing the random seed to use for reproducibility. Default is 0
    Returns
    -------
    new_masks: List containing all the masks permuted globally
    """

    layer_wise_flatten = []                      # maintain the layerwise flattened tensor
    for i in range(len(old_masks)):
        layer_wise_flatten.append(old_masks[i].flatten())

    global_flatten = []
    for i in range(len(layer_wise_flatten)):
        if len(global_flatten) == 0:
            global_flatten.append(layer_wise_flatten[i].cpu())
        else:
            global_flatten[-1] = np.append(global_flatten[-1], layer_wise_flatten[i].cpu())
    permuted_mask = np.random.permutation(global_flatten[-1])
        
    new_masks = []
    idx1 = 0
    idx2 = 0
    for i in range(len(old_masks)):
        till_idx = old_masks[i].numel()
        idx2 = idx2 + till_idx
        new_masks.append(permuted_mask[idx1:idx2].reshape(old_masks[i].shape))
        idx1 = idx2

    # Convert to tensor
    for i in range(len(new_masks)):
        new_masks[i] = torch.tensor(new_masks[i])

    return new_masks

def original_initialization(model, mask_temp, initial_state_dict):
    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0
    return model


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    else:
        print("pruning/weight_init() Not supported nn.modules"+str(m))
