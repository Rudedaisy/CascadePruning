import sys
sys.path.insert(1, '/root/hostCurUser/attention-is-all-you-need-pytorch')
import train as transformer_train
import translate as transformer_translate
from transformer.Models import Transformer
import vgg16 as vgg16_old
#from resnet import *
from models import vgg16, vgg_in, resnet, resnet_in, inception_v3, inception_v3_c10, alexnet
import torch
import torch.nn as nn
import numpy as np
from pruned_layers import *

import pickle

sys.path.insert(1, '/root/hostCurUser/attention-is-all-you-need-pytorch')
import CSP.pruned_layers

import argparse
import matplotlib.pyplot as plt
import math
from operator import truediv
import statistics

from prune import prune

skipLinearExport = True

#parser = argparse.ArgumentParser(description='Quick analysis')
#parser.add_argument('--path', type=str, default='', help='file to load pretrained weights from')
#parser.add_argument('--chunk', type=int, default=32, help='chunk size (ie systolic array width)')
#args = parser.parse_args()

#assert (args.path != '')

EXPORT_PATH = "/root/hostCurUser/reproduce/DNNsim/net_traces/VGG16_CIFAR10_CSP/"
#EXPORT_PATH = "foo/"
MODEL_PATH = "/root/hostCurUser/reproduce/DNNsim/models/VGG16_CIFAR10_CSP/"
#MODEL_PATH = "foo/"

SCALE_PATH = "/root/hostCurUser/root/SCALE-Sim-cascade/topologies/conv_nets/VGG16_CIFAR10.csv"
#SCALE_PATH = "/root/hostCurUser/root/SCALE-Sim-cascade/topologies/transformer/Transformer.csv"
CX_PATH = "/root/hostCurUser/reproduce/cambriconx/data/vgg16/cifar10/"
#CX_PATH = "foo/"
ST_PATH = "/root/hostCurUser/reproduce/SparTen/data/"
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

def replace_with_pruned(m, name, skipLinear=False):
    if type(m) == PrunedConv or type(m) == PrunedLinear:
        return
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.Conv2d:
            print("Replaced CONV")
            setattr(m, attr_str, PrunedConv(target_attr))
        elif type(target_attr) == torch.nn.Linear and (not skipLinear):
            print("Replaced Linear")
            setattr(m, attr_str, PrunedLinear(target_attr))

    for n, ch in m.named_children():
        replace_with_pruned(ch, n, skipLinear)

def replace_vgg16(model, skipLinear=False):
    for i in range(len(model.features)):
        print(model.features[i])
        if isinstance(model.features[i], torch.nn.Conv2d):
            print("Replaced CONV")
            model.features[i] = PrunedConv(model.features[i])
    for i in range(len(model.classifier)):
        if isinstance(model.classifier[i], torch.nn.Linear) and (not skipLinear):
            print("Replaced Linear")
            model.classifier[i] = PrunedLinear(model.classifier[i])
    return model
        
def exportData(pathName, modelName):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if modelName=="vgg16_c10":
        model = vgg16.VGG16()
    elif modelName=="vgg16_c10_old":
        model = vgg16_old.VGG16()
    elif modelName=="resnet50_c10":
        model = resnet.ResNet50()
    elif modelName=="inception_v3_c10":
        model = inception_v3_c10.inception_v3()
    elif modelName=="alexnet_in":
        #model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
        model = alexnet.AlexNet()
        skipLinear = True
    elif modelName=="vgg16_in":
        model = vgg_in.vgg16()
        skipLinear = True
    elif modelName=="resnet50_in":
        model = resnet_in.resnet50()
    elif modelName=="transformer_wmt":
        translator = transformer_translate.create_model() # returns Transformer()
        translator = translator.to(device)
        model =	translator.model
    else:
        print("Model not supported")
        exit(1)

    model = model.to(device)

    if ("vgg16" in modelName or "alexnet" in modelName):
        model = replace_vgg16(model, skipLinear)
    elif "transformer" in modelName:
        #transformer_train.replace_with_pruned(model, "model", prune_attention=True, prune_only_attention=False)
        pass
    else:
        replace_with_pruned(model, "model", skipLinear)

    if "_in" in modelName:
        if not torch.distributed.is_initialized():
            port = np.random.randint(10000, 65536)
            torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d'%port, rank=0, world_size=1)
        model = torch.nn.parallel.DistributedDataParallel(model)

    if not ("transformer" in modelName):
        model.load_state_dict(torch.load(pathName))
    model.eval()
    
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
    layer_idx = 0

    models = []  # for SparTen
    layers = []
    acts = []
    weights = []
    paddings = []
    strides = []
    idxs = []
    
    def extract(module, input):
        #if module.extracted:
        #    return
        if len(input[0].shape) < 4:
            if not ("transformer" in modelName):
                return
            try:
                a = input[0].detach().cpu().reshape(1, module.in_features, 1, 1)
            except:
                a = input[0].detach().cpu().reshape(-1, 1, 1)
                a = a[:module.in_features]
                a = a.reshape(1, module.in_features, 1, 1)
        else:
            a = input[0].detach().cpu()
        acts.append(a)

        if isinstance(module, torch.nn.Conv2d):
            layer = module.weight.view((module.out_channels, -1)).detach().cpu().numpy()
            weight = module.weight.detach().cpu().numpy()
            tp = "conv"
            stride = str(max(module.stride[0], module.stride[1]))
            padding = str(max(module.padding[0], module.padding[1]))
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            padding_st = module.padding
            stride_st = module.stride
        elif isinstance(module, torch.nn.Linear) and (not skipLinearExport):
            layer = module.weight.view((module.out_features, -1)).detach().cpu().numpy()
            weight = module.weight.detach().cpu().reshape(module.weight.shape[0], module.weight.shape[1], 1, 1).numpy()
            tp = "fc"
            stride = str(1)
            padding = str(0)
            in_channels = module.in_features
            out_channels = module.out_features
            kernel_size = (1,1)
            padding_st = (0,0)
            stride_st = (1,1)
        else:
            print("{} does not exist".format(module))
            exit(1)
        name = str(module.layer_idx)
        weights.append(weight)
        paddings.append(int(padding))
        strides.append(int(stride))
        f.write(name+"," + tp+"," + stride+"," + padding + ",\n")
        layers.append(layer)
        models.append({'in_channels': in_channels,
                       'out_channels': out_channels,
                       'kernel': kernel_size,
                       'name': tp+name,
                       'padding': padding_st,
                       'weights': weight,
                       'IFM': a.cpu().numpy(),
                       'stride': stride_st
        })
        idxs.append(module.layer_idx)
        #module.extracted = True
        
    #replace_with_exporter(model, "model", f, fscale)
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            weight = m.weight.detach().cpu().numpy()
            if weight.shape[2] != weight.shape[3]:
                continue
            #weights.append(weight)
            #m.extracted = False
            m.register_forward_pre_hook(extract)
            #name = str(layer_idx)
            #tp = "conv"
            #stride = str(max(m.stride[0], m.stride[1]))
            #padding = str(max(m.padding[0], m.padding[1]))
            #paddings.append(int(padding))
            #f.write(name+"," + tp+"," + stride+"," + padding + ",\n")
            m.layer_idx = layer_idx
            layer_idx += 1
        elif isinstance(m, torch.nn.Linear):
            if not ("transformer" in modelName):
                continue
            #weight = m.weight.detach().cpu().reshape(m.weight.shape[0], m.weight.shape[1], 1, 1).numpy()
            #weights.append(weight)
            #m.extracted = False
            m.register_forward_pre_hook(extract)
            #name = str(layer_idx)
            #tp = "fc"
            #stride = str(1)
            #padding = str(0)
            #paddings.append(int(padding))
            #f.write(name+"," + tp+"," + stride+"," + padding + ",\n")
            m.layer_idx = layer_idx
            layer_idx += 1

    if "_in" in modelName:
        IFM = torch.rand(1, 3, 224, 224).cuda()
        model(IFM)
    elif "_c10" in modelName:
        IFM = torch.rand(1, 3, 32, 32).cuda()
        model(IFM)
    elif "_wmt" in modelName:
        src_seq = [4556, 4562, 4560, 4557, 4712, 1894, 15, 4564, 4620, 0, 5]
        pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
        print(pred_seq)
    else:
        print("Dataset not supported")
        exit(1)

    print(len(acts))
    print(len(weights))
        
    #layers = loadCoeffIdx(pathName, modelName)

    with open(ST_PATH+modelName+".h5", "wb") as f:
        pickle.dump(models, f)
    
    sq_ptrs = []
    out_channels = []
    for layer in layers:
        sp, oc = squeezeCoeffIdxTotal(layer, len(layer))
        out_channels.append(oc)
    #"""
    i = 0
    for idx in range(len(acts)):
        x_save = acts[idx]
        weight_save = weights[idx]
        np.save(EXPORT_PATH + "act-" + str(idx) + "-0.npy", x_save)
        np.save(EXPORT_PATH + "wgt-" + str(idx) + ".npy", weight_save)

        x_save[x_save == 0] = int(0)
        x_save[x_save != 0] = int(1)
        np.savetxt(CX_PATH + "act"+str(idx)+".csv", x_save.reshape(-1), delimiter=",", fmt="%d")
        weight_save[weight_save == 0] = int(0)
        weight_save[weight_save != 0] = int(1)
        np.savetxt(CX_PATH + "Conv2D_"+str(idx)+".csv", weight_save.reshape(weight_save.shape[0], -1), delimiter=",", fmt="%d")
        
        if x_save.shape[2] > 1:
            name = "Conv" + str(idx) #str(idxs[idx])
            IFM_height = str(x_save.shape[2] + (2*paddings[idx]))
            IFM_width = str(x_save.shape[3] + (2*paddings[idx]))
            filt_height = str(weight_save.shape[2])
            filt_width = str(weight_save.shape[3])
        else:
            name = "FC" + str(idx) #str(idxs[idx])
            IFM_height = str(1)
            IFM_width = str(1)
            filt_height = str(1)
            filt_width = str(1)
        channels = str(weight_save.shape[1])
        if ("resnet50" in modelName) and (idx == 4 or idx == 15 or idx == 29 or idx == 49):
            num_filt = str(weight_save.shape[0])
        else:
            num_filt = str(out_channels[i])
            i += 1
        fscale.write(name+',\t' + IFM_height+',\t' + IFM_width+',\t' + filt_height+',\t' + filt_width+',\t' + channels+',\t' + num_filt+',\t' + str(strides[idx])+',\n')

    fscale.close()
    f.close()
    #"""    
    cxf = open(CX_PATH + "ModelShape.txt", "w")
    cxf.write("LayerName\tLayerID\tInputShape\tOutputShape\tKernelShape\n")
    cx_layers = []
    layer_idx = 0
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.weight.shape[2] != m.weight.shape[3]:
                continue
            curr = "Conv\t" + str(layer_idx)+"\t" + str(tuple(acts[layer_idx].shape)).replace('(','').replace(')','').replace(' ','')+"\t" + str(tuple(acts[layer_idx].shape)).replace('(','').replace(')','').replace(' ','')+"\t" + str(tuple(m.weight.shape)).replace('(','').replace(')','').replace(' ','')+"\n"
            cxf.write(curr)
            cx_layers.append(curr)
            layer_idx += 1
        if isinstance(m, torch.nn.Linear) and (not skipLinear):
            curr = "FC\t" + str(idxs[layer_idx])+"\t" + str(tuple(acts[layer_idx].shape)).replace('(','').replace(')','').replace(' ','')+"\t" + str(tuple(acts[layer_idx].shape)).replace('(','').replace(')','').replace(' ','')+"\t" + str(tuple(m.weight.shape)).replace('(','').replace(')','').replace(' ','')+",1,1\n"
            cxf.write(curr)
            cx_layers.append(curr)
            layer_idx += 1
    if "transformer" in modelName:
        for i in range(11):
            for j in range(36,97):
                cxf.write(cx_layers[j])
    cxf.close()

    return
    
def loadCoeffIdx(pathName, modelName):
    skipLinear = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if modelName=="vgg16_c10":
        model = vgg16.VGG16()
    elif modelName=="vgg16_c10_old":
        model = vgg16_old.VGG16()
    elif modelName=="resnet50_c10":
        model = resnet.ResNet50()
    elif modelName=="inceptionv3_c10":
        model = inception_v3_c10.inception_v3()
    elif modelName=="alexnet_in":
        #model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
        model =	alexnet.AlexNet()
        skipLinear = True
    elif modelName=="vgg16_in":
        model = vgg_in.vgg16()
    elif modelName=="resnet50_in":
        model = resnet_in.resnet50()
    elif modelName=="transformer_wmt":
        translator = transformer_translate.create_model() # returns Transformer()
        translator = translator.to(device)
        model = translator.model
    else:
        print("Model not supported")
        exit(1)

    model = model.to(device)

    if ("vgg16" in modelName or "alexnet" in modelName):
        model = replace_vgg16(model, skipLinear)
    elif "transformer" in modelName:
        #transformer_train.replace_with_pruned(model, "model", prune_attention=True, prune_only_attention=False)
        pass
    else:
        replace_with_pruned(model, "model", skipLinear)

    if "_in" in modelName:
        if not torch.distributed.is_initialized():
            port = np.random.randint(10000, 65536)
            torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d'%port, rank=0, world_size=1)
        model = torch.nn.parallel.DistributedDataParallel(model)

    if not ("transformer" in modelName):
        model.load_state_dict(torch.load(pathName))
    model.eval()

    #prune(model, method="cascade", q=1.0)
    
    layers = []
    widths = [3]
    numConv = 0
    assert isinstance(model, nn.Module)
    for n, m in model.named_modules():
        print(type(m))
        if isinstance(m, torch.nn.Conv2d):
            numConv += 1
        if isinstance(m, torch.nn.Linear):
            numConv += 1
        if isinstance(m, PrunedConv):
            #print(m.mask)
            #layer = m.mask.view((m.out_channels, -1)).detach().cpu().numpy()
            layer = m.conv.weight.view((m.out_channels, -1)).detach().cpu().numpy()
        elif isinstance(m, PrunedLinear) or isinstance(m, CSP.pruned_layers.PrunedLinear) and (not skipLinear):
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

    if "transformer" in modelName:
        for i in range(11):
            for j in range(36,97):
                layers.append(layers[j])
    
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

    nonzero_subrows = sum(sq_ptrs)
    tot_subrows = layer.shape[1] * num_fold
    
    return sq_ptrs #, nonzero_subrows, tot_subrows

def squeezeCoeffIdxTotal(layer, array_w):
    sq_ptrs = []
    out_channels = []
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

    # squeeze the ineffectual output channels
    out_channels = len(np.where(layer.any(axis=1))[0])
    print(str(out_channels) + "<" + str(layer.shape[0]))
        
    return sq_ptrs, out_channels

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

    #PATH = '/root/hostCurUser/root/CascadePruning/ckpt/ResNet03270957/retrain_weight_73_0.94.pt'
    PATH = '/root/hostCurUser/root/CascadePruning/ckpt/VGG1603261948/retrain_weight_38_0.92.pt'
    #PATH = '/root/hostCurUser/root/CascadePruning/ckpt/Inception304112240/retrain_weight_74_0.94.pt'
    #PATH = '/root/hostCurUser/root/SCALE-Sim-cascade/topologies/conv_nets/VGG16_1e-4_5e-4_ft_q0.75_1e-4_5e-4_70.89.pt'
    #PATH = '/root/hostCurUser/root/SCALE-Sim-cascade/topologies/conv_nets/AlexNet_1e-4_1e-4_ft_q1.0_1e-4_5e-4_55.62.pt'
    #PATH = '/root/hostCurUser/root/CascadePruning/ckpt/DistributedDataParallel04052011/retrain_weight_27_69.35.pt'
    #PATH = '/root/hostCurUser/attention-is-all-you-need-pytorch/output/pruned_attention2/finetuned_model.chkpt'
    
    #layers = loadCoeffIdx(PATH)
    #nonzeros = 0
    #tot = 0
    #for layer in layers:
    #    #print(len(cascadeCompress(layer, 32)))
    #    sq_ptrs, nonzero_subrows, tot_subrows = squeezeCoeffIdx(layer, 32)
    #    nonzeros += nonzero_subrows
    #    tot += tot_subrows
    #print("Chunk sparsity: {}".format(1 - (float(nonzeros) / tot)))
        
    exportData(PATH, "vgg16_c10")
    
if __name__ == "__main__":
    main()
