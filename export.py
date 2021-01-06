from vgg16 import VGG16, VGG16_half, VGG16_5
from resnet import *
import torch
import torch.nn as nn
import numpy as np
from pruned_layers import *

import argparse
import matplotlib.pyplot as plt
import math
from operator import truediv
import statistics

#parser = argparse.ArgumentParser(description='Quick analysis')
#parser.add_argument('--path', type=str, default='', help='file to load pretrained weights from')
#parser.add_argument('--chunk', type=int, default=32, help='chunk size (ie systolic array width)')
#args = parser.parse_args()

#assert (args.path != '')

def loadCoeffIdx(pathName):
    device1 = 'cuda' if torch.cuda.is_available() else 'cpu'
    #net = ResNet50()
    net1 = VGG16()
    net1 = net1.to(device)

    net1.load_state_dict(torch.load(pathName))

    layers = []
    widths = [3]
    assert isinstance(net1, nn.Module)
    for n, m in net1.named_modules():
        if isinstance(m, PrunedConv):
            layer = m.conv.weight.view((m.out_channels, -1)).detach().cpu().numpy()
        elif isinstance(m, PruneLinear):
            layer = m.linear.weight.view((m.out_features, -1)).detach().cpu().numpy()
        else:
            continue
        
        #print(n)
        #print(layer.shape)

        #layer = layer > 0
        widths.append(len(layer))
        layers.append(layer)
    
        #layerMask = layer > 0
        #layerMask = np.transpose(layerMask)
    return layers
    
def squeezeCoeffIdx(layer, array_w):    
    sq_ptrs = []
    #layer = layers[layerIdx]
    layer = layer > 0
    num_fold = math.ceil(len(layer) / array_w)
    
    for chunk_idx in range(num_fold):
        sq_ptrs.append(0)
        if (chunk_idx+1) * array_w >= len(layer):
            maxL = len(layer)
        else:
            maxL = (chunk_idx+1) * array_w

        chunk = layer[chunk_idx * array_w:(chunk_idx * array_w + maxL)]
        sq_ptrs[chunk_idx] = len(np.where(chunk.any(axis=0))[0])

    return sq_ptrs

def simplify(layers):
    """ simplify weights to 1's and 0's """
    for layer_idx in range(len(layers)):
        layers[layer_idx] = (layers[layer_idx].astype(bool)).astype(int)
    return layers

def cascadeCompress(layer, array_w):
    # apply 'simplify' function here as a sanity check
    #layer = simplify([layer])[0]
    ptrs = []
    num_fold = math.ceil(len(layer) / array_w)

    for row_idx in range(len(layer[0])):
        ptrs.append(0) # numChunksInRow
        for chunk_idx in range(num_fold):
            if (chunk_idx+1) * array_w >= len(layer):
                if not any(layer[chunk_idx*array_w:len(layer), row_idx]):
                    break
            else:
                if not any(layer[chunk_idx*array_w:(chunk_idx+1)*array_w, row_idx]):
                    break
            ptrs[-1] += 1

    return ptrs
        

def main():
    #accelerator_ranges = [0, 2, 6, 14, 30, 62, 126]
    
    #PATH = 'ckpt/finetune_VGG1609131701/pruned_weight_92_0.90.pt'
    PATH = 'ckpt/finetune_VGG1609061144/pruned_weight_63_0.90.pt'
    layers = loadCoeffIdx(PATH)
    #layers = simplify(layers)
    for layer in layers:
        print(len(cascadeCompress(layer, 32)))
        #print(squeezeCoeffIdx(layer, 32, None, None))
        
if __name__ == "__main__":
    main()
