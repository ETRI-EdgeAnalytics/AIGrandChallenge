import torch
import numpy as np

def binarize(n, index_bits=4):
    ret = []
    for b in np.binary_repr(n, index_bits):
        ret.append(b=='1')
    return np.array(ret, dtype=np.uint8) 
    
def decimalize(b):
    d = 0
    len_ = len(b)
    for idx, val in enumerate(b):
        d += (val == 1) * pow(2, len_ - idx - 1)
    return d

def pack(indices, index_bits=4):
    temp = np.array([], dtype=np.uint8)
    bit_acc = 0
    last_idx = -1
    packed_array = np.array([], dtype=np.uint8)
    for i, idx in enumerate(indices):
        # idx -> ridx 
        if last_idx == -1:
            ridx = idx
        else:
            ridx = idx - last_idx - 1
        last_idx = idx

        # split ridx into index_bits-values
        ridx_ = ridx
        while ridx_ >= pow(2, index_bits) - 1:
            ridx_ -= pow(2, index_bits) - 1
            temp = np.append(temp, np.array([1] * index_bits, dtype=np.uint8))
            if len(temp) % 8 == 0: # unit8. push the value into the array.
                packed_array = np.append(packed_array, np.packbits(temp, axis=None))
                bit_acc += len(temp)
                temp = np.array([], dtype=np.uint8)

        b = binarize(ridx_, index_bits)
        temp = np.append(temp, b)
        if len(temp) % 8 == 0 or i == len(indices) - 1 :
            packed_array = np.append(packed_array, np.packbits(temp, axis=None))
            bit_acc += len(temp)
            temp = np.array([], dtype=np.uint8)
    return packed_array, bit_acc

def unpack(packed_array, index_bits, tbits):
    unpacked = np.unpackbits(packed_array, axis=None, count=tbits)

    indices = []
    last_idx = -1

    bit_cnt = 0
    while bit_cnt < tbits:
        if bit_cnt + index_bits > tbits:
            index_bits_ = tbits % index_bits
        else:
            index_bits_ = index_bits

        b = unpacked[bit_cnt:bit_cnt+index_bits_]
        bit_cnt += index_bits
        d = decimalize(b)

        if d == pow(2, index_bits) - 1: # padded
            last_idx += d
        else:
            indices.append(last_idx+1+d)
            last_idx = indices[-1]
    return indices

def sparse_format(flatten_w):
    indices = []
    values = []
    for i, v in enumerate(flatten_w):
        if v != 0:
            indices.append(i)
            values.append(v)
    return indices, values

def dense_format(indices, values, t, len_):
    flatten_w = np.zeros(len_)
    for i, v in zip(indices, values):
        flatten_w[i] = v
    return flatten_w

def dump_weights(filename, weights, index_bits=8, val_post=lambda x: x):
    weights_ = {}
    for name, weight in weights.items():
        indices, values = sparse_format(weight.numpy().flatten())
        weights_[name] = [pack(indices, index_bits=index_bits), val_post(values)]
    np.savez_compressed(filename, w=weights_)

def dump(filename, model, items=None, index_bits=8, val_post=lambda x: x):
    if items is None:
        items = ["weight", "bias"]
    weights = {}
    for name, module in model.named_modules():
        for key in items:
            if hasattr(module, key) and getattr(module, key) is not None:
                weights[name + "." +key] = getattr(module, key).data
    dump_weights(filename, weights, index_bits, val_post)

def load_weights(filename, shapes, types, index_bits=8, val_pre=lambda x: x):
    weights_ = np.load(filename, allow_pickle=True)["w"].item()
    weights = {}
    for name, weight in weights_.items():
        indices = unpack(weights_[name][0][0], index_bits=index_bits, tbits=weights_[name][0][1])
        values = val_pre(weights_[name][1])
        if name not in types:
            weights[name] = dense_format(indices, values, np.float32, np.prod(shapes[name]))
        else:
            weights[name] = dense_format(indices, values, types[name], np.prod(shapes[name]))
        weights[name] = np.reshape(weights[name], shapes[name])
    return weights
 
def load(filename, model, items=None, index_bits=8, val_pre=lambda x: x):
    if items is None:
        items = ["weight", "bias"]
    shapes = {}
    types = {}
    for name, module in model.named_modules():
        for key in items:
            if hasattr(module, key) and getattr(module, key) is not None:
                shapes[name+"."+key] = getattr(module, key).data.shape
                types[name+"."+key] = getattr(module, key).data.dtype

    weights = load_weights(filename, shapes, types, index_bits, val_pre)
    for name, module in model.named_modules():
        for key in items:
            if hasattr(module, key) and getattr(module, key) is not None:
                getattr(module, key).data = torch.tensor(weights[name+"."+key], requires_grad=True, dtype=types[name+"."+key])

def main():
    from rexnetv1 import ReXNetV1
    model = ReXNetV1()
    model.eval()
    data = torch.rand((5, 3, 224, 224))
    y = model(data)
    dump("test.npz", model)
    load("test.npz", model)
    y2 = model(data)

    print(y, y2)
    print(y == y2)

    print("done")
    """
    import random
    for i in range(10000):
        size = int(random.random() * 1000)+10
        arr = []
        for j in range(size):
            arr.append(int(random.random() * 10000))
        x = sorted(list(set(arr)))

        arr, acc = pack(x, 4)
        x_ = unpack(arr, 4, acc)
        print(x, x_)
        assert x == x_
    """

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/jongryul/work/system/challenges/GrandChallenge/model")
    main()
