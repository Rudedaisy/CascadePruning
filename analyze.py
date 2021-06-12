from vgg16 import VGG16, VGG16_half, VGG16_5
from resnet import *
from models import vgg16, vgg_in, resnet, resnet_in, inception_v3, inception_v3_c10
import torch
import torch.nn as nn
import numpy as np
from pruned_layers import *

import argparse
import matplotlib.pyplot as plt
import math
from operator import truediv
import statistics

parser = argparse.ArgumentParser(description='Quick analysis')
parser.add_argument('--path', type=str, default='', help='file to load pretrained weights from')
parser.add_argument('--chunk', type=int, default=32, help='chunk size (ie systolic array width)')
args = parser.parse_args()

assert (args.path != '')

def replace_with_pruned(m, name):
    if type(m) == PrunedConv or type(m) == PrunedLinear:
        return
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.Conv2d:
            print("Replaced CONV")
            setattr(m, attr_str, PrunedConv(target_attr))
        elif type(target_attr) == torch.nn.Linear:
            print("Replaced Linear")
            setattr(m, attr_str, PrunedLinear(target_attr))

    for n, ch in m.named_children():
        replace_with_pruned(ch, n)

def replace_vgg16(model):
    for i in range(len(model.features)):
        print(model.features[i])
        if isinstance(model.features[i], torch.nn.Conv2d):
            print("Replaced CONV")
            model.features[i] = PrunedConv(model.features[i])
    for i in range(len(model.classifier)):
        if isinstance(model.classifier[i], torch.nn.Linear):
            print("Replaced Linear")
            model.classifier[i] = PrunedLinear(model.classifier[i])
    return model
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = resnet.ResNet50()
#model = resnet_in.resnet50()
#model = vgg16.VGG16()
#model = inception_v3_c10.inception_v3()
model = model.to(device)

replace_with_pruned(model, "model")
#model = replace_vgg16(model)

#if not torch.distributed.is_initialized():
#    port = np.random.randint(10000, 65536)
#    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d'%port, rank=0, world_size=1)
#model = torch.nn.parallel.DistributedDataParallel(model)

model.load_state_dict(torch.load(args.path))

layers = []
widths = [3]
assert isinstance(model, nn.Module)
for n, m in model.named_modules():
    if isinstance(m, PrunedConv):
        layer = m.conv.weight.view((m.out_channels, -1)).detach().cpu().numpy()
    elif isinstance(m, PrunedLinear):
        layer = m.linear.weight.view((m.out_features, -1)).detach().cpu().numpy()
    else:
        continue
    
    #print(n)
    #print(layer.shape)

    widths.append(len(layer))
    layers.append(layer)
    
    layerMask = layer > 0
    layerMask = np.transpose(layerMask)
"""
    # print info
    #YB = range(m.out_channels)
    #XB = range(len(layer))
    #X,Y = np.meshgrid(XB,YB)
    #plt.imshow(layer,interpolation='none')
    plt.imsave('images/' + str(n) + '.png', layerMask)
    #break
"""
sq_ptrs = []
full_ptrs = []
ratios = []
#chunk_track = []
total_ptrs = []
first_row_IFM_accesses = 0
for idx, layer in enumerate(layers):
    sq_ptrs.append([])
    full_ptrs.append([])
    ratios.append([])
    num_fold = math.ceil(len(layer) / args.chunk)
    layer = layer > 0

    #chunk_track.append(num_fold)
    for chunk_idx in range(num_fold):
        sq_ptrs[idx].append(0)
        full_ptrs[idx].append(len(layer[0]))
        if (chunk_idx+1) * args.chunk >= len(layer):
            maxL = len(layer)
        else:
            maxL = (chunk_idx+1) * args.chunk

        chunk = layer[chunk_idx * args.chunk:(chunk_idx * args.chunk + maxL)]
        sq_ptrs[idx][chunk_idx] = len(np.where(chunk.any(axis=0))[0])

        if chunk_idx == 0:
            first_row_IFM_accesses += len(np.where(chunk.any(axis=0))[0])

        if chunk_idx >= len(total_ptrs):
            total_ptrs.append(0)
        total_ptrs[chunk_idx] += len(np.where(chunk.any(axis=0))[0])

        if chunk_idx == 0:
            ratios[idx].append(sq_ptrs[idx][chunk_idx] / len(layer[0]))
        elif sq_ptrs[idx][chunk_idx-1] != 0:
            ratios[idx].append(sq_ptrs[idx][chunk_idx] / sq_ptrs[idx][chunk_idx-1])
        else:
            ratios[idx].append(0)
            
#print(sq_ptrs)
#print(full_ptrs)

accelerator_ranges = [0, 2, 6, 14, 30, 62, 126]

weightedMean = 0
accelerator_ptrs = []
for idx in range(len(total_ptrs)):
    weightedMean += total_ptrs[idx] * (idx+1)

    for i in range(len(accelerator_ranges)):
        if idx >= accelerator_ranges[i] and idx < accelerator_ranges[i+1]:
            acc_idx = i
            break
    if acc_idx >= len(accelerator_ptrs):
        accelerator_ptrs.append(0)
    accelerator_ptrs[acc_idx] += float(total_ptrs[idx])
m = float(max(accelerator_ptrs))
alpha = 0
numBins = 3
for idx in range(len(accelerator_ptrs)):
    print(str(accelerator_ptrs[idx]) + "/" + str(m))
    accelerator_ptrs[idx] = float(accelerator_ptrs[idx]) / m
    alpha += (accelerator_ptrs[idx] * ((accelerator_ranges[idx+1] - accelerator_ranges[idx]) / accelerator_ranges[numBins]))
for idx in range(len(total_ptrs)-1):
    total_ptrs[idx] -= total_ptrs[idx+1]

print(total_ptrs)
print(accelerator_ptrs)

exit(0)

print("Chunk count statistics")
print("ALPHA for dynamic power metric: {}".format(alpha))
print("Mean: {}".format(weightedMean))
print("First row IFM accesses: {}".format(first_row_IFM_accesses))
#print(chunk_track)
#print("Mean: {}".format(sum(total_ptrs) / len(total_ptrs)))
#print("Std: {}".format(statistics.pstdev(total_ptrs)))
#print("Max: {}".format(max(total_ptrs)))
    
fig,ax = plt.subplots()
ax.plot(total_ptrs)
ax.set_xlabel('Chunk count - 1', fontsize='large')
ax.set_ylabel('Frequency', fontsize='large')
ax.grid(True)
#plt.yscale('log')
plt.ylim(0.0)
plt.savefig('images/totalChunks.png')

fig,ax = plt.subplots()
ax.plot(accelerator_ptrs)
ax.set_xlabel('Register bin', fontsize='large')
ax.set_ylabel('Frequency', fontsize='large')
ax.grid(True)
#plt.yscale('log')
plt.savefig('images/acceleratorChunks.png')

exit(0)

full = list(map(truediv, full_ptrs[14], full_ptrs[14]))
fig, ax = plt.subplots()
ax.plot(full)
keys = ['Baseline']
for idx in range(len(sq_ptrs)-3):
    if idx <= 1:
        color = 'tab:red'
    elif idx <= 3:
        color = 'tab:blue'
    elif idx <= 6:
        color = 'tab:green'
    else:
        color = 'tab:orange'
    curr = list(map(truediv, sq_ptrs[idx], full_ptrs[idx]))
    ax.plot(curr, color=color)
    keys.append('Pruned layer ' + str(idx))
    
ax.set_xlabel('Chunk idx', fontsize='large')
ax.set_ylabel('% of elements', fontsize='large')
#ax.legend(keys, bbox_to_anchor=(1.00, 1))
ax.grid(True)
#plt.ylim(0.0,1.0)
plt.savefig('images/chunkWidths.png')
    

for idx in range(len(sq_ptrs)):
    """
    fig, ax = plt.subplots()
    ax.plot(full_ptrs[idx])
    ax.plot(sq_ptrs[idx])
    ax.set_xlabel('Chunk idx', fontsize='large')
    ax.set_ylabel('Number of elements', fontsize='large')
    ax.legend(['Baseline', 'Pruned Chunks'])
    ax.grid(True)
    plt.ylim(bottom=0)
    plt.savefig('images/chunkWidths' + str(idx) + '.png')

    fig, ax = plt.subplots()
    ax.plot(ratios[idx])
    ax.set_xlabel('Chunk idx', fontsize='large')
    ax.set_ylabel('Ratio to prev. layer', fontsize='large')
    ax.grid(True)
    plt.ylim(0.0,1.0)
    plt.savefig('images/ratios' + str(idx) + '.png')
    """
    fig, ax = plt.subplots()                                                                                                           

    color = 'tab:red'
    ax.plot(full_ptrs[idx], color=color)
    ax.plot(sq_ptrs[idx], color=color)
    ax.set_xlabel('Chunk idx', fontsize='large')
    ax.set_ylabel('Number of elements', fontsize='large', color=color)
    ax.legend(['Baseline', 'Pruned Chunks'])
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='y', labelcolor=color)

    ax2 = ax.twinx()
    color = 'tab:blue'
    ax2.plot(ratios[idx], color=color)
    #ax2.set_xlabel('Chunk idx', fontsize='large')
    ax2.set_ylabel('Ratio to prev. layer', fontsize='large', color=color)
    ax2.set_ylim(0.0,1.0)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.savefig('images/combined' + str(idx) + '.png')
