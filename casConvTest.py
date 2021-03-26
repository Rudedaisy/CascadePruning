#This works under my own filer_decomposition repo
import torch
from pruned_layers import *
from CasConv import casConv2d
from models import utils
#from train_util import test
from models import vgg16
#from vgg16 import VGG16

model = vgg16.VGG16()
#model = VGG16()
model.cuda()

#"""
for i in range(len(model.features)):
    if isinstance(model.features[i], torch.nn.Conv2d):
        model.features[i] = PrunedConv(model.features[i])
    if isinstance(model.features[i], torch.nn.Linear):
        model.features[i] = PrunedLinear(model.features[i])
#"""

#model.load_state_dict(torch.load("./ckpt/baseline/VGG16_93.49.pt"))
#model.load_state_dict(torch.load("./ckpt/finetune_VGG1609061144/pruned_weight_63_0.90.pt"))
#model.load_state_dict(torch.load("./ckpt/finetune_VGG1601261725/pruned_weight_48_0.89.pt"))
#model.load_state_dict(torch.load("./ckpt/VGG1603061343/retrain_weight_73_0.93.pt"))
model.load_state_dict(torch.load("./ckpt/VGG1603091405/retrain_weight_48_0.92.pt"))
model.cuda()

"""
for quant in range(8,1,-1):
    print("---------QUANTIZE: {}---------".format(quant))
    for ars in range(2,34,2):
        
        for i in range(len(model.features)):
            if isinstance(model.features[i], torch.nn.Conv2d):
                model.features[i] = casConv2d(model.features[i], array_size=ars, quant=quant)
            elif isinstance(model.features[i], PrunedConv):
                model.features[i] = casConv2d(model.features[i].conv, array_size=ars, quant=quant)
            elif isinstance(model.features[i], casConv2d):
                model.features[i] = casConv2d(model.features[i].conv2d_module, array_size=ars, quant=quant)
        
        print("Array size: {}".format(ars))
        utils.eval_cifar10(model, 32)
        #test("CIFAR10", model)
"""

for quant in range(8,1, -1):
    print("---------QUANTIZE: {}---------".format(quant))
    for i in range(len(model.features)):                                                                                                                                                  
        if isinstance(model.features[i], torch.nn.Conv2d):                                                                                                                                
            model.features[i] = casConv2d(model.features[i], array_size=1, quant=quant)                                                                                                 
        elif isinstance(model.features[i], PrunedConv):                                                                                                                                     
            model.features[i] = casConv2d(model.features[i].conv, array_size=1, quant=quant)                                                                                            
        elif isinstance(model.features[i], casConv2d):                                                                                                                                      
            model.features[i] = casConv2d(model.features[i].conv2d_module, array_size=1, quant=quant)
    utils.eval_cifar10(model, 32)
