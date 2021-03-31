"""
ReXNet
Copyright (c) 2020-present NAVER Corp.
MIT license

<Weight Pruning, Quantization-Aware Training Version, ETRI>
"""

import torch
import torch.nn as nn
from math import ceil

from torch.quantization import QuantStub, DeQuantStub
from helper_func import *

from rexnetv1_quant import *

saved_model_dir = './'

float_model_file = 'rexnet_pretrained_float.pth'
scripted_float_model_file = 'rexnet_quantization_scripted.pth'
scripted_quantized_model_file = 'rexnet_quantization_scripted_quantized.pth'
scripted_quantized_per_channel_model_file = 'rexnet_quantization_scripted_quantized_per_channel.pth'
scripted_quantized_qat_model_file = 'rexnet_quantization_scripted_quantized_qat.pth'

data_path = './imagenet_41'
train_batch_size = 30
eval_batch_size = 30
num_calibration_batches = 10
num_eval_batches = 10
data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()

init_model = ReXNetV1(width_mult=2.0, classes=41)
torch.save(init_model.state_dict(), saved_model_dir + float_model_file)

def load_model(model_file):
    model = ReXNetV1(width_mult=2.0, classes=41)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

#####################################################################################
# baseline
float_model = load_model(saved_model_dir + float_model_file).to('cpu')
float_model.eval()

# Fuses modules
float_model.fuse_model()

print_size_of_model(float_model)
top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

#####################################################################################

#####################################################################################
# 4. Post-training static quantization
# Post-training static quantization

myModel = load_model(saved_model_dir + float_model_file).to('cpu')
myModel.eval()

# Fuse Conv, bn and relu
myModel.fuse_model()

criterion = nn.CrossEntropyLoss()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
myModel.qconfig = torch.quantization.default_qconfig
print(myModel.qconfig)
torch.quantization.prepare(myModel, inplace=True)
# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')

# Calibrate with the training set
evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(myModel, inplace=True)
print('Post Training Quantization: Convert done')

print("Size of model after quantization")
print_size_of_model(myModel)
top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(myModel), saved_model_dir + scripted_quantized_per_channel_model_file)

# For this quantized model, we see a significantly lower accuracy of just ~62% on these same 300
# images. Nevertheless, we did reduce the size of our model down to just under 3.6 MB, almost a 4x
# decrease.
#
# In addition, we can significantly improve on the accuracy simply by using a different
# quantization configuration. We repeat the same exercise with the recommended configuration for
# quantizing for x86 architectures. This configuration does the following:
#
# - Quantizes weights on a per-channel basis
# - Uses a histogram observer that collects a histogram of activations and then picks
#   quantization parameters in an optimal manner.
#

per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
per_channel_quantized_model.eval()
per_channel_quantized_model.fuse_model()
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(per_channel_quantized_model.qconfig)

torch.quantization.prepare(per_channel_quantized_model, inplace=True)
evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
torch.quantization.convert(per_channel_quantized_model, inplace=True)
top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_per_channel_model_file)


#####################################################################################


#####################################################################################
# 5. Quantization-aware training
# ------------------------------
#
# Quantization-aware training (QAT) is the quantization method that typically results in the highest accuracy.
# With QAT, all weights and activations are “fake quantized” during both the forward and backward passes of
# training: that is, float values are rounded to mimic int8 values, but all computations are still done with
# floating point numbers. Thus, all the weight adjustments during training are made while “aware” of the fact
# that the model will ultimately be quantized; after quantizing, therefore, this method will usually yield
# higher accuracy than either dynamic quantization or post-training static quantization.
#
# The overall workflow for actually performing QAT is very similar to before:
#
# - We can use the same model as before: there is no additional preparation needed for quantization-aware
#   training.
# - We need to use a ``qconfig`` specifying what kind of fake-quantization is to be inserted after weights
#   and activations, instead of specifying observers
#
# We first define a training function:

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return

# We fuse modules as before
qat_model = load_model(saved_model_dir + float_model_file)
qat_model.fuse_model()

optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# Finally, ``prepare_qat`` performs the "fake quantization", preparing the model for quantization-aware
# training

torch.quantization.prepare_qat(qat_model, inplace=True)

# Training a quantized model with high accuracy requires accurate modeling of numerics at
# inference. For quantization aware training, therefore, we modify the training loop by:
#
# - Switch batch norm to use running mean and variance towards the end of training to better
#   match inference numerics.
# - We also freeze the quantizer parameters (scale and zero-point) and fine tune the weights.

num_train_batches = 20

# Train and check accuracy after each epoch
best_top1 = 0.0
for nepoch in range(20):
    train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
    if nepoch > 3:
        # Freeze quantizer parameters
        qat_model.apply(torch.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
    quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
    quantized_model.eval()
    top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
    if best_top1 < top1.avg:
        print('Saving qat model(best_top1: %2.2f <  %2.2f , %s)' %(best_top1, top1.avg,scripted_quantized_qat_model_file))
        best_top1 = top1.avg
        torch.jit.save(torch.jit.script(quantized_model), saved_model_dir + scripted_quantized_qat_model_file)
    print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))

print("Size of model after quantization-aware training")
print_size_of_model(quantized_model)


#####################################################################
# Here, we just perform quantization-aware training for a small number of epochs. Nevertheless,
# quantization-aware training yields an accuracy of over 71% on the entire imagenet dataset,
# which is close to the floating point accuracy of 71.9%.
#
# More on quantization-aware training:
#
# - QAT is a super-set of post training quant techniques that allows for more debugging.
#   For example, we can analyze if the accuracy of the model is limited by weight or activation
#   quantization.
# - We can also simulate the accuracy of a quantized model in floating point since
#   we are using fake-quantization to model the numerics of actual quantized arithmetic.
# - We can mimic post training quantization easily too.
#
# Speedup from quantization
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Finally, let's confirm something we alluded to above: do our quantized models actually perform inference
# faster? Let's test:

def run_benchmark(hdr, model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)

    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms (%s)' % (elapsed/num_images*1000, hdr))
    print('Size (MB):', os.path.getsize(model_file)/1e6)
    top1, top5 = evaluate(model,criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy : %2.2f'%( top1.avg))
    print('===============================')

    return elapsed

run_benchmark("floating-point model",saved_model_dir + scripted_float_model_file, data_loader_test)

run_benchmark("quantized,per channel model",saved_model_dir + scripted_quantized_per_channel_model_file, data_loader_test)

run_benchmark("qat model",saved_model_dir + scripted_quantized_qat_model_file, data_loader_test)



