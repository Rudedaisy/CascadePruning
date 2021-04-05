from vgg16 import VGG16, VGG16_half, VGG16_5
#from resnet import *
from models import vgg16, vgg_in, resnet, resnet_in, inception_v3
import torch
import torch.nn as nn
import numpy as np
from pruned_layers import *

import argparse
import matplotlib.pyplot as plt
import math
from operator import truediv
import statistics

from prune import prune

#parser = argparse.ArgumentParser(description='Quick analysis')
#parser.add_argument('--path', type=str, default='', help='file to load pretrained weights from')
#parser.add_argument('--chunk', type=int, default=32, help='chunk size (ie systolic array width)')
#args = parser.parse_args()

#assert (args.path != '')

EXPORT_PATH = "/root/hostCurUser/reproduce/DNNsim/net_traces/ResNet50_ImageNet_CSP/"
MODEL_PATH = "/root/hostCurUser/reproduce/DNNsim/models/ResNet50_ImageNet_CSP/"
SCALE_PATH = "/root/hostCurUser/root/SCALE-Sim-cascade/topologies/conv_nets/ResNet50_CIFAR10.csv"
LAYER_IDX = 0

class DataExporter(nn.Module):
    def __init__(self, module, save_dir, layer_idx, f, fscale):
        super(DataExporter, self).__init__()
        assert (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)), "DataExporter needs either Conv2d or Linear module"

        self.module = module
        self.save_dir = save_dir
        self.name = str(layer_idx)
        self.f = f
        self.fscale = fscale
        
    def forward(self, x):
        out = self.module(x)

        if isinstance(self.module, nn.Conv2d):
            name = "Conv" + self.name
            tp = "conv"
            stride = str(self.module.stride[0])
            padding = str(self.module.padding[0])
        else:
            name = "FC" + self.name
            tp = "fc"
            stride = str(1)
            padding = str(0)
        
        print("Saving {}".format(name))
        
        # Save IFM
        x_save = x.detach().cpu().numpy()
        np.save(self.save_dir + "act-" + name + "-0.npy", x_save)
        
        # Save weights
        weight_save = self.module.weight.detach().cpu().numpy()
        np.save(self.save_dir + "wgt-" + name + ".npy", weight_save)

        self.f.write(name+"," + tp+"," + stride+"," + padding + ",\n")

        # SCALE-Sim config ----
        if isinstance(self.module, nn.Conv2d):
            IFM_height = str(x_save.shape[2] + (2*self.module.padding[0]))
            IFM_width = str(x_save.shape[3] + (2*self.module.padding[1]))
            filt_height = str(weight_save.shape[2])
            filt_width = str(weight_save.shape[3])
        else:
            IFM_height = str(1)
            IFM_width = str(1)
            filt_height = str(1)
            filt_width = str(1)
        channels = str(weight_save.shape[1])
        num_filt = str(weight_save.shape[0])
        
        self.fscale.write(name+',\t' + IFM_height+',\t' + IFM_width+',\t' + filt_height+',\t' + filt_width+',\t' + channels+',\t' + num_filt+',\t' + stride+',\n')
        
        return out

def replace_with_exporter(m, name, f, fscale):
    global LAYER_IDX
    if type(m) == DataExporter:
        return
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == nn.Conv2d or type(target_attr) == nn.Linear:
            print("Replaced with exporter")
            setattr(m, attr_str, DataExporter(target_attr, EXPORT_PATH, LAYER_IDX, f, fscale))
            LAYER_IDX += 1
            
    for n, ch in m.named_children():
        replace_with_exporter(ch, n, f, fscale)

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

def exportData(pathName):
    device1 = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model = resnet_in.resnet50()
    model = resnet.ResNet50()
    model = model.to(device)

    replace_with_pruned(model, "model")

    #if not torch.distributed.is_initialized():
    #    port = np.random.randint(10000, 65536)
    #    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d'%port, rank=0, world_size=1)
    #model = torch.nn.parallel.DistributedDataParallel(model)

    model.load_state_dict(torch.load(pathName))
    
    #prune(model, method="cascade", q=1.0)

    #layer_num = 0
    #save_dir = "/root/hostCurUser/reproduce/DNNsim/net_traces/ResNet50_ImageNet_CSP/"
    #lstModules = list( model.named_modules())
    #for n, m in model.named_modules():
    #for i in range(len(lstModules)):
    #    if isinstance(lstModules[i], nn.Conv2d) or isinstance(lstModules[i], nn.Linear):
    #        model[i] = DataExporter(lstModules[i], save_dir, layer_num)
    #        layer_num += 1


    f = open(MODEL_PATH + "model.csv", "w")
    fscale = open(SCALE_PATH, "w")
    fscale.write("Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,\n")
    replace_with_exporter(model, "model", f, fscale)

    #for n, m in model.named_modules():
    #    print(m)
    
    #IFM = torch.rand(1, 3, 224, 224).cuda()
    IFM = torch.rand(1, 3, 32, 32).cuda()
    model(IFM)
    fscale.close()
    f.close()

    return
    
def loadCoeffIdx(pathName):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet.ResNet50()
    #model = VGG16()
    #model = resnet_in.resnet50()
    model = model.to(device)

    replace_with_pruned(model, "model")

    #if not torch.distributed.is_initialized():
    #    port = np.random.randint(10000, 65536)
    #    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d'%port, rank=0, world_size=1)
    #model = torch.nn.parallel.DistributedDataParallel(model)
    
    model.load_state_dict(torch.load(pathName))

    #prune(model, method="cascade", q=1.0)
    
    layers = []
    widths = [3]
    numConv = 0
    assert isinstance(model, nn.Module)
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            numConv += 1
        if isinstance(m, torch.nn.Linear):
            numConv += 1
        if isinstance(m, PrunedConv):
            #print(m.mask)
            #layer = m.mask.view((m.out_channels, -1)).detach().cpu().numpy()
            layer = m.conv.weight.view((m.out_channels, -1)).detach().cpu().numpy()
        elif isinstance(m, PrunedLinear):
            #print(m.mask)
            #layer = m.mask.view((m.out_features, -1)).detach().cpu().numpy()
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
    #print("NUM CONV: {}".format(numConv))
    #print("NUM EXTRACTED LAYERS: {}".format(len(layers)))
    return layers
    
def squeezeCoeffIdx(layer, array_w):    
    sq_ptrs = []
    #layer = layers[layerIdx]
    layer = layer > 0
    #print(layer)
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
    #PATH = 'ckpt/finetune_VGG1609061144/pruned_weight_63_0.90.pt'
    #PATH = '/root/hostCurUser/root/CascadePruning/ckpt/DistributedDataParallel03202251/retrain_weight_88_76.85.pt'

    #PATH = "ckpt/VGG1603082154/retrain_weight_41_0.92.pt"
    #PATH = "ckpt/finetune_VGG1601261725/pruned_weight_48_0.89.pt"

    PATH = '/root/hostCurUser/root/CascadePruning/ckpt/ResNet03270957/retrain_weight_73_0.94.pt'
    #PATH = '/root/hostCurUser/root/CascadePruning/ckpt/ResNet03262148/retrain_weight_90_0.94.pt'
    
    #layers = loadCoeffIdx(PATH)
    
    #for layer in layers:
    #    print(len(cascadeCompress(layer, 32)))
    #    #print(squeezeCoeffIdx(layer, 32, None, None))

    #exportData(PATH)
    layers = loadCoeffIdx(PATH)
    numWeights = 0
    for layer in layers:
        layer = layer > 0
        numWeights += np.sum(layer)
    print(numWeights)
    
if __name__ == "__main__":
    main()
