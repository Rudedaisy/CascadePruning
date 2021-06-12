from models import vgg16, vgg_in, resnet, resnet_in, inception_v3, inception_v3_c10
from decompose.decomConv import DecomposedConv2D
from pruned_layers import *
import numpy as np
import pickle
import torch
import torch.nn as nn

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
        
def export_sparten(modelpath, input_shape, filename):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model = vgg16.VGG16()
    #model = resnet.ResNet50()
    #model = inception_v3_c10.inception_v3()
    model = resnet_in.resnet50()
    model = model.to(device)
    
    replace_with_pruned(model, "model")
    #model = replace_vgg16(model)

    if not torch.distributed.is_initialized():
        port = np.random.randint(10000, 65536)
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d'%port, rank=0, world_size=1)
    model = torch.nn.parallel.DistributedDataParallel(model)
    
    model.load_state_dict(torch.load(modelpath))
    
    #model = torch.load(modelpath)
    sample_in = torch.randn(input_shape).cuda()
    models = []

    model.eval()
    acts = []

    count = 0
    def extract(module, input):
        if len(input[0].shape) < 4:
            assert False
            act = input[0].detach().reshape(1, input[0].shape[1], 1, 1)
        else:
            act = input[0].detach()
        #acts.append(input[0].detach())
        acts.append(act)
        print(act.shape)

    last_output = sample_in
    for n, m in model.named_modules():
        """
        if isinstance(m, DecomposedConv2D):
            weight = torch.mm(m.coefs.reshape(m.out_channels * m.in_channels, m.num_basis), m.basis)\
                .view((m.out_channels, m.in_channels, m.kernel_size[0], m.kernel_size[1]))
            models.append({'in_channels': m.in_channels,
                           'out_channels': m.out_channels,
                           'kernel': m.kernel_size[0],
                           'name': n,
                           'padding': m.padding[0],
                           'weights': weight.detach().cpu().numpy(),
                           'IFM': [],
                           'stride': m.stride})
            m.register_forward_pre_hook(extract)
        elif isinstance(m, torch.nn.Conv2d):
            weight = m.weight
            models.append({'in_channels': m.in_channels,
                           'out_channels': m.out_channels,
                           'kernel': m.kernel_size[0],
                           'name': n,
                           'padding': m.padding[0],
                           'coefs': weight.detach().cpu().numpy(),
                           'IFM': [],
                           'stride': m.stride})
            m.register_forward_pre_hook(extract)
        """
        if isinstance(m, torch.nn.Conv2d):
            weight = m.weight
            models.append({'in_channels': m.in_channels,
                           'out_channels': m.out_channels,
                           'kernel': m.kernel_size[0],
                           'name': n,
                           'padding': m.padding[0],
                           'weights': weight.detach().cpu().numpy(),
                           'IFM': [],
                           'stride': m.stride})
            print(weight.shape)
            m.register_forward_pre_hook(extract)
            count += 1
        #elif isinstance(m, torch.nn.Linear):
        #    weight = m.weight
        #    models.append({'in_channels': m.in_features,
        #                   'out_channels': m.out_features,
        #                   'kernel': 1,
        #                   'name': n,
        #                   'padding': 0,
        #                   'weights': weight.detach().cpu().reshape(weight.shape[0], weight.shape[1], 1, 1).numpy(),
        #                   'IFM': [],
        #                   'stride': (1,1)})
        #    m.register_forward_pre_hook(extract)
        #    count += 1
            
    model(sample_in)
    total = 0
    for i in range(len(acts)):
        models[i]['IFM'] = acts[i].cpu().numpy()
        try:
            total += acts[i].shape[2] * acts[i].shape[3] * models[i]['out_channels']
        except:
            assert False
            total += acts[i].shape[1] * models[i]['out_channels']
    print(total)

    with open(filename, 'wb') as f:
        pickle.dump(models, f)
    print("Count: {}".format(count))

filename = "/root/hostCurUser/reproduce/SparTen/data/ResNet50_ImageNet_SparTen.h5"
#modelpath = "/root/hostCurUser/root/CascadePruning/ckpt/ResNet03270957/retrain_weight_73_0.94.pt"
#modelpath = "/root/hostCurUser/root/CascadePruning/ckpt/VGG1603261948/retrain_weight_38_0.92.pt"
#modelpath = "/root/hostCurUser/root/CascadePruning/ckpt/Inception304112240/retrain_weight_74_0.94.pt"
modelpath = "/root/hostCurUser/root/CascadePruning/ckpt/DistributedDataParallel04052011/retrain_weight_27_69.35.pt"
input_shape = (1, 3, 224, 224)
export_sparten(modelpath, input_shape, filename)
