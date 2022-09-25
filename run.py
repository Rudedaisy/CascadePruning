#from vgg16 import VGG16, VGG16_half, VGG16_5
#from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
#from train_util import train, train_cifar10, test, test_cifar10
from summary import summary
import torch
import numpy as np
from prune import prune
from pruned_layers import *
from NM_pruned_layers import NMSparseConv, NMSparseLinear
import argparse
import sys

import pickle

import argparse
import random
import numpy as np

from models import utils
from models import vgg16, vgg_in, resnet, resnet_in, inception_v3, inception_v3_c10, mobilenetv3, fc_mnist
from models import helpers
from models.extract import export
from models.efficientnet.model import EfficientNet
from models.efficientnet.utils import Conv2dStaticSamePadding
from models.MLP import MLP

# Importing individual images for model extraction
#import larq_zoo as lqz
#from urllib3.request import urlopen
#from PIL import Image


parser = argparse.ArgumentParser(description='Bounded Structured Sparsity')

parser.add_argument('--skip-pt', action='store_true', default=False, help='skip pretrain and simply load weights directly')
parser.add_argument('--path', type=str, default='', help='file to load pretrained weights from')
parser.add_argument('--model', type=str, default='vgg16', help='model to use, options: [vgg16, resnet18, resnet50, inception_v3, alexnet, mlp, fc_mnist]')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset to train on: [MNIST, CIFAR10, ImageNet]')
parser.add_argument("--data-dir", type=str, default='/root/hostPublic/ImageNet/', help="the path to dataset folder.")

parser.add_argument('--ckpt-dir', type=str, default='', help='checkpoint save/load directory, default=ckpt/<modelName><time>/')
parser.add_argument('--epochs', type=int, default=100, help='pretrain number of epochs, default=100')
parser.add_argument('--batch', type=int, default=128, help='pretrain and finetune batch size, default=128')
parser.add_argument('--lr', type=float, default=0.01, help='pretrain initial learning rate, default=0.01')
parser.add_argument('--reg', type=float, default=5e-4, help='pretrain reg strength, default=5e-4')
parser.add_argument('--scheduler', type=str, default='cosine', help="scheduler ['step', 'cosine','coswm']")
parser.add_argument('--spar-reg', type=str, default='v2', help='sparsity regularizer type, options: [None, v1, v2, SSL, NM]')
parser.add_argument('--spar-str', type=float, default=1e-4, help='sparsity reg strength, default=1e-4')
parser.add_argument('--scratch', action='store_false', default=True, help='train from scratch, default=False[ImageNet]')

parser.add_argument('--prune', action='store_true', default=False, help='apply prune, then finetune -- WARNING: must be a separate process from the initial pre-/re-training stage!')
parser.add_argument('--chunk-size', type=int, default=32, help='chunk size of structural pruning methods')
parser.add_argument('--prune-type', type=str, default='cascade', help='pruning scheme, options: [percentage, std, dil, asym_dil, sintf, chunk, cascade, SSL, cs]')
parser.add_argument('--q', type=float, default=0, help='prune threshold, will default to prune-type\'s default if not specified')

parser.add_argument('--ckpt-dir-ft', type=str, default='', help='finetune checkpoint directory, default=ckpt/finetune_<modelName><time>/')
parser.add_argument('--epochs-ft', type=int, default=50, help='finetune number of epochs, default=50')
parser.add_argument('--lr-ft', type=float, default=0.001, help='finetune initial learning rate, default=0.001')
parser.add_argument('--reg-ft', type=float, default=5e-6, help='finetune reg strength, default=5e-6')

parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument('--extract', action='store_true', default=False, help='Extract IFM and weight data from all relevant layers')
parser.add_argument('--extract_backward', action='store_true', default=False, help='Extract input and weight gradients from backward pass')

args = parser.parse_args()

if args.spar_reg == "NM":
    fup = True
else:
    fup = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.set_deterministic(True)

set_seed(42)
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

world_size = torch.distributed.get_world_size()
local_rank = torch.distributed.get_rank()

device = torch.device('cuda', args.local_rank)
print(device)

#Scale single process batch_size and learning rate accordingly
eff_bs = args.batch // world_size
eff_lr = args.lr * np.sqrt(world_size)
eff_lr_ft = args.lr_ft * np.sqrt(world_size)

assert ((not args.skip_pt) or (args.path != ''))

if args.q == 0:
    if args.prune_type == 'percentage':
        args.q = 45.0
    elif args.prune_type == 'std':
        args.q = 0.75
    elif args.prune_type == 'sintf':
        args.q = 0.75
    elif args.prune_type == 'chunk':
        args.q = 0.75
    elif args.prune_type == 'cascade':
        args.q = 0.75

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------------------------- #
# --- Full precision model load/train --- #
# --------------------------------------- #

if args.dataset == "MNIST" and args.model != "fc_mnist":
    print(f"Dataset {args.dataset} not supported for model {args.model}")
    exit(1)
if args.model == "vgg16":
    if args.dataset == "ImageNet":
        model = vgg_in.vgg16(pretrained=args.scratch)
    else:
        model = vgg16.VGG16()
elif args.model == "resnet18":
    if args.dataset == "ImageNet":
        model = resnet_in.resnet18(pretrained=args.scratch)
    else:
        model = resnet.ResNet18()
elif args.model == "resnet50":
    if args.dataset == "ImageNet":
        model = resnet_in.resnet50(pretrained=args.scratch)
    else:
        model = resnet.ResNet50()
elif args.model == "resnet152":
    if args.dataset == "ImageNet":
        model = resnet_in.resnet152(pretrained=args.scratch)
    else:
        print("Model {} not supported!".format(args.model))
        sys.exit(0)
elif args.model == "inception_v3":
    if args.dataset == "ImageNet":
        model = inception_v3.gluon_inception_v3(pretrained=args.scratch)
    else:
        model = inception_v3_c10.inception_v3()
elif args.model == "alexnet":
    if args.dataset == "ImageNet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=args.scratch)
        #Remove dropout for the retraining
        #model.classifier[0] = torch.nn.Identity()
    else:
        print("Model {} not supported!".format(args.model))
        sys.exit(0)
elif args.model == "efficientnet":
    model = EfficientNet.from_pretrained('efficientnet-b1')
elif args.model == 'mbnetv3':
    model = mobilenetv3.mobilenetv3_large()
    model.load_state_dict(torch.load("ckpt/mobilenetv3-large-1cd25616.pth"))
elif args.model == "mlp":
    if args.dataset == "ImageNet":
        print(f"Model {args.model} not supported!")
    else:
        model = MLP()
elif args.model == "fc_mnist":
    if args.dataset == "MNIST":
        model = fc_mnist.FC_MNIST()
    else:
        print(f"Model {args.model} not supported!")
else:
    print("Model {} not supported!".format(args.model))
    sys.exit(0)
model = model.to(device)

#for layer in model.named_modules():
#    print(layer)
    
def replace_with_pruned(m, name, spar_reg):    
    #print(m)
    print("{}, {}".format(name, str(type(m))))
    if type(m) == PrunedConv or type(m) == PrunedLinear or type(m) == NMSparseConv or type(m) == NMSparseLinear:
        return

    # HACK: directly replace conv layers of downsamples
    if name == "downsample":
        if spar_reg == "NM":
            m[0] = NMSparseConv(m[0])
        else:
            m[0] = PrunedConv(m[0], args.chunk_size)
    
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.Conv2d or type(target_attr) == Conv2dStaticSamePadding:
            print("Replaced CONV")
            if spar_reg == "NM":
                setattr(m, attr_str, NMSparseConv(target_attr))
            else:
                setattr(m, attr_str, PrunedConv(target_attr, args.chunk_size))
        elif type(target_attr) == torch.nn.Linear:
            print("Replaced Linear")
            if spar_reg == "NM":
                setattr(m, attr_str, NMSparseLinear(target_attr))
            else:
                setattr(m, attr_str, PrunedLinear(target_attr, args.chunk_size))

    for n, ch in m.named_children():
        replace_with_pruned(ch, n, spar_reg)


if args.spar_reg != "None":
    if args.model != "vgg16" and args.model != "alexnet":
        replace_with_pruned(model, "model", args.spar_reg)
    else:
        for i in range(len(model.features)):
            print(model.features[i])
            if isinstance(model.features[i], torch.nn.Conv2d):
                print("Replaced CONV")
                if args.spar_reg == "NM":
                    model.features[i] = NMSparseConv(model.features[i])
                else:
                    model.features[i] = PrunedConv(model.features[i], args.chunk_size)
    #for i in range(len(model.classifier)):
    #    if isinstance(model.classifier[i], torch.nn.Linear):
    #        print("Replaced Linear")
    #        model.classifier[i] = PrunedLinear(model.classifier[i], args.chunk_size)

if local_rank == 0:
        
    for layer in model.named_modules():
        print(layer)
print("WARNING (for CSP reg): only CONV layers are targetted for pruning")

model = model.to(device)

if args.extract_backward:
    print("Extracting forward and backward passes")
    if args.dataset == "ImageNet":
        train_func = utils.train_imagenet
        train_dict = {'model':model, 'epochs':args.epochs, 'batch_size':eff_bs, 'lr':eff_lr, 'reg':args.reg, 'device':device,
                      'checkpoint_path':args.ckpt_dir, 'spar_reg':args.spar_reg, 'spar_param':args.spar_str,
                      'scheduler':args.scheduler, 'data_dir':args.data_dir, 'finetune':False, 'amp':True, 'lmdb':True, 'find_unused_parameters':fup, 'extract':True}
    elif args.dataset == "CIFAR10":
        train_func = utils.train_cifar10
        train_dict = {'model':model, 'epochs':args.epochs, 'batch_size':args.batch, 'lr':args.lr, 'reg':args.reg,
                      'checkpoint_path':args.ckpt_dir, 'spar_reg':args.spar_reg, 'spar_param':args.spar_str,
                      'scheduler':'step', 'finetune':False, 'cross':False, 'cross_interval':5, 'extract':True}
        #elif args.dataset == "MNIST":
        #    train_func = utils.train_mnist
    else:
        print(f"ERR: dataset {args.dataset} not supported")
        exit(1)
    export(model, args.model, "extract/", train_func, backwards=True, train_dict=train_dict)
    exit(0)
elif args.extract:
    print("Extracting model. No training.")
    if args.dataset == "ImageNet":
        #IFM = torch.rand(1, 3, 224, 224).cuda()
        # Want a small batch size of 3 images
        inference_func = utils.val_imagenet
    elif args.dataset == "CIFAR10":
        inference_func = utils.eval_cifar10
    elif args.dataset == "MNIST":
        inference_func = utils.eval_mnist
    else:
        print(f"ERR: dataset {args.dataset} not supported")
        exit(1)
    export(model, args.model, "extract/", inference_func)
    exit(0)

# Uncomment to load pretrained weights
#model.load_state_dict(torch.load("model_before_pruning.pt"))

# Comment if you have loaded pretrained weights
# Tune the hyperparameters here.
#train(model, epochs=35, batch_size=128, lr=0.01, reg=0.005)
#train(model, epochs=60, batch_size=128, lr=0.01, reg=1e-4, spar_reg='v2', spar_param=1e-4, checkpoint_path=args.ckpt_dir)
if not args.skip_pt:
    if args.dataset=="MNIST":
        utils.train_mnist(model, epochs=args.epochs, batch_size=args.batch, lr=args.lr, reg=args.reg,
                          checkpoint_path=args.ckpt_dir, spar_reg=args.spar_reg, spar_param=args.spar_str,
                          finetune=False, cross=False, cross_interval=5)
    elif args.dataset=="CIFAR10":
        utils.train_cifar10(model, epochs=args.epochs, batch_size=args.batch, lr=args.lr, reg=args.reg,
                            checkpoint_path = args.ckpt_dir, spar_reg = args.spar_reg, spar_param = args.spar_str,
                            scheduler='step', finetune=False, cross=False, cross_interval=5)
    elif args.dataset == "ImageNet":
        utils.train_imagenet(model, epochs=args.epochs, batch_size=eff_bs, lr=eff_lr, reg=args.reg, device=device,
                             checkpoint_path = args.ckpt_dir, spar_reg = args.spar_reg, spar_param = args.spar_str,
                             scheduler=args.scheduler, data_dir=args.data_dir, finetune=False, amp=True, lmdb=True, find_unused_parameters=fup)
    else:
        print("Dataset {} not suported!".format(args.dataset))
        sys.exit(0)
else:
    #if args.dataset == "ImageNet":
    #    if not torch.distributed.is_initialized():
    #        port = np.random.randint(10000, 65536)
    #        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d'%port, rank=0, world_size=1)
    #    model = torch.nn.parallel.DistributedDataParallel(model)

    model.load_state_dict(torch.load(args.path, map_location=device),)
    #helpers.load_state_dict(args.path)
    print("Model loaded from {}".format(args.path))

print("-----Summary before pruning-----")
summary(model)
print("-------------------------------")

pickle.dump(model, open("foo.pkl","wb"))

if not args.prune:
    print("Option to prune and finetune not chosen. Exiting")
    sys.exit(0)
if args.spar_reg == "NM":
    print("No pruning/finetuning necessary with NM sparsity")
    sys.exit(0)

# --------------------------------------- #
# --- Pruning and finetune -------------- #
# --------------------------------------- #

# Test accuracy before fine-tuning
prune(model, method=args.prune_type, q=args.q)
if args.dataset=="CIFAR10":
    utils.eval_cifar10(model, batch_size=128)
    #test(args.dataset, model)
elif args.dataset == "ImageNet":
    #utils.val_imagenet(model, lmdb=True, amp=True)
    print("Val_imagenet not working!")
else:
    print("Dataset {} not suported!".format(args.dataset))
    sys.exit(0)

print("-----Summary After pruning-----")
summary(model)
print("-------------------------------")

# Uncomment to load pretrained weights
#net.load_state_dict(torch.load("net_after_pruning.pt"))
# Comment if you have loaded pretrained weights
#finetune_after_prune(model, epochs=50, batch_size=128, lr=0.01, reg=0.005)
if args.dataset=="CIFAR10":
    #train(args.dataset, model, finetune=True, epochs=args.epochs_ft, batch_size=args.batch, lr=args.lr_ft, reg=args.reg_ft, checkpoint_path=(args.ckpt_dir_ft))
    utils.train_cifar10(model, epochs=args.epochs_ft, batch_size=args.batch, lr=args.lr_ft, reg=args.reg_ft,
                        checkpoint_path = args.ckpt_dir_ft, spar_reg = args.spar_reg, spar_param = args.spar_str,
                        scheduler='step', finetune=True, cross=False, cross_interval=5)
elif args.dataset == "ImageNet":
    utils.train_imagenet(model, epochs=args.epochs_ft, batch_size=eff_bs, lr=eff_lr_ft, reg=args.reg, device=device,
                         checkpoint_path = args.ckpt_dir, spar_reg = args.spar_reg, spar_param = args.spar_str,
                         scheduler=args.scheduler, data_dir=args.data_dir, finetune=True, amp=True, lmdb=True, find_unused_parameters=fup)
else:
    print("Dataset {} not suported!".format(args.dataset))
    sys.exit(0)

summary(model)
