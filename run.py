#from vgg16 import VGG16, VGG16_half, VGG16_5
#from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
#from train_util import train, train_cifar10, test, test_cifar10
from summary import summary
import torch
import numpy as np
from prune import prune
from pruned_layers import *
import argparse
import sys

from models import utils
from models import vgg16, resnet

parser = argparse.ArgumentParser(description='Bounded Structured Sparsity')

parser.add_argument('--skip-pt', action='store_true', default=False, help='skip pretrain and simply load weights directly')
parser.add_argument('--path', type=str, default='', help='file to load pretrained weights from')
parser.add_argument('--model', type=str, default='vgg16', help='model to use, options: [vgg16, resnet50]')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset to train on: [CIFAR10, ImageNet]')

parser.add_argument('--ckpt-dir', type=str, default='', help='checkpoint save/load directory, default=ckpt/<modelName><time>/')
parser.add_argument('--epochs', type=int, default=100, help='pretrain number of epochs, default=100')
parser.add_argument('--batch', type=int, default=128, help='pretrain and finetune batch size, default=128')
parser.add_argument('--lr', type=float, default=0.01, help='pretrain initial learning rate, default=0.01')
parser.add_argument('--reg', type=float, default=5e-4, help='pretrain reg strength, default=5e-4')
parser.add_argument('--spar-reg', type=str, default='v2', help='sparsity regularizer type, options: [None, v1, v2, SSL]')
parser.add_argument('--spar-str', type=float, default=1e-4, help='sparsity reg strength, default=1e-4')

parser.add_argument('--prune-type', type=str, default='cascade', help='pruning scheme, options: [percentage, std, dil, asym_dil, sintf, chunk, cascade, SSL]')
parser.add_argument('--q', type=float, default=0, help='prune threshold, will default to prune-type\'s default if not specified')

parser.add_argument('--ckpt-dir-ft', type=str, default='', help='finetune checkpoint directory, default=ckpt/finetune_<modelName><time>/')
parser.add_argument('--epochs-ft', type=int, default=50, help='finetune number of epochs, default=50')
parser.add_argument('--lr-ft', type=float, default=0.001, help='finetune initial learning rate, default=0.001')
parser.add_argument('--reg-ft', type=float, default=5e-6, help='finetune reg strength, default=5e-6')

args = parser.parse_args()

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------------------------- #
# --- Full precision model load/train --- #
# --------------------------------------- #

if args.model == "vgg16":
    model = vgg16.VGG16()
elif args.model == "resnet50":
    model = resenet.ResNet50()
else:
    print("Model {} not supported!".format(args.model))
    sys.exit(0)
model = model.to(device)

for i in range(len(model.features)):
    if isinstance(model.features[i], torch.nn.Conv2d):
        model.features[i] = PrunedConv(model.features[i])
    if isinstance(model.features[i], torch.nn.Linear):
        model.features[i] = PrunedLinear(model.features[i])

# Uncomment to load pretrained weights
#model.load_state_dict(torch.load("model_before_pruning.pt"))

# Comment if you have loaded pretrained weights
# Tune the hyperparameters here.
#train(model, epochs=35, batch_size=128, lr=0.01, reg=0.005)
#train(model, epochs=60, batch_size=128, lr=0.01, reg=1e-4, spar_reg='v2', spar_param=1e-4, checkpoint_path=args.ckpt_dir)
if not args.skip_pt:
    #train(args.dataset, model, finetune=False, epochs=args.epochs, batch_size=args.batch, lr=args.lr, reg=args.reg, spar_reg=args.spar_reg, spar_param=args.spar_str, checkpoint_path=args.ckpt_dir)
    if args.dataset=="CIFAR10":
        utils.train_cifar10(model, epochs=args.epochs, batch_size=args.batch, lr=args.lr, reg=args.reg,
                            checkpoint_path = args.ckpt_dir, spar_reg = args.spar_reg, spar_param = args.spar_str,
                            scheduler='step', finetune=False, cross=False, cross_interval=5)
    else:
        print("Dataset {} not suported!".format(args.dataset))
        sys.exit(0)
else:
    model.load_state_dict(torch.load(args.path))
    print("Model loaded from {}".format(args.path))

print("-----Summary before pruning-----")
summary(model)
print("-------------------------------")

#sys.exit(0) ########## REMOVE IF PRUNING AND FINETUNEING

# --------------------------------------- #
# --- Pruning and finetune -------------- #
# --------------------------------------- #

# Test accuracy before fine-tuning
prune(model, method=args.prune_type, q=args.q)
if args.dataset=="CIFAR10":
    utils.eval_cifar10(model, batch_size=128)
    #test(args.dataset, model)
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
else:
    print("Dataset {} not suported!".format(args.dataset))
    sys.exit(0)
