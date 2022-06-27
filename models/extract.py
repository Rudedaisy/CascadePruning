import math
import torch
import torch.nn as nn
import pickle
import numpy as np
from colorama import Fore
from colorama import Style
from colorama import init

""" --- Do not change --- """
LAYER_IDX = 0
init(autoreset=True)
""" --------------------- """

def export(model, model_name, out_path, inference_func):
    global LAYER_IDX
    
    model.eval()
    models = []

    def extract(module, input):
        global LAYER_IDX

        if isinstance(module, torch.nn.Conv2d) or issubclass(type(module), torch.nn.Conv2d):
            layer = module.weight.view((module.out_channels, -1)).detach().cpu().numpy()
            if "get_sparse_weights" in dir(module):
                weight = module.get_sparse_weights().detach().cpu().numpy()
            else:
                weight = module.weight.detach().cpu().numpy()
            tp = "conv"
            stride = str(max(module.stride[0], module.stride[1]))
            padding = str(max(module.padding[0], module.padding[1]))
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = (weight.shape[2], weight.shape[3])
            padding = module.padding
            stride = module.stride
        elif isinstance(module, torch.nn.Linear) or issubclass(type(module), torch.nn.Linear):
            layer = module.weight.view((module.out_features, -1)).detach().cpu().numpy()
            if "get_sparse_weights" in dir(module):
                weight = module.get_sparse_weights().detach().cpu().numpy()
            else:
                weight = module.weight.detach().cpu().reshape(module.weight.shape[0], module.weight.shape[1], 1, 1).numpy()
            tp = "fc"
            stride = str(1)
            padding = str(0)
            in_channels = module.in_features
            out_channels = module.out_features
            kernel_size = (1,1)
            padding = (0,0)
            stride = (1,1)
        else:
            print(Style.DIM + "{} found. Ignored.".format(type(module)))
            return
        if len(input[0].shape) < 4:
            # FC layer
            a = input[0].detach().cpu().reshape(-1, module.in_features, 1, 1)
        else:
            # CONV layer
            a = input[0].detach().cpu()
            
        print(Fore.GREEN + "Recording {} with shape {}-{}-{}. IFM shape {}.".format(tp+str(LAYER_IDX), out_channels, in_channels, kernel_size, a.shape))
        models.append({'in_channels': in_channels,
                       'out_channels': out_channels,
                       'kernel': kernel_size,
                       'name': tp+str(LAYER_IDX),
                       'padding': padding,
                       'weights': weight,
                       'IFM': a.cpu().numpy(),
                       'stride': stride
        })
        LAYER_IDX += 1
        
    for n, m in model.named_modules():
        m.register_forward_pre_hook(extract)

    #model(IFM)
    inference_func(model, lmdb=True, amp=True, extract=True)
    
    with open(out_path+model_name+".h5", "wb") as f:
        pickle.dump(models, f)

    #print(models)
    print("Finished exporting to {}".format(out_path+model_name+".h5"))
            
if __name__ == '__main__':
    export(in_path=IN_PATH, model_name=MODEL_NAME, out_path=OUT_PATH)

    
