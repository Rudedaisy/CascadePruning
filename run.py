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

import argparse
import random
import numpy as np

from models import utils
from models import vgg16, vgg_in, resnet, resnet_in, inception_v3, inception_v3_c10, mobilenetv3
from models import helpers
from models.efficientnet.model import EfficientNet

parser = argparse.ArgumentParser(description='Bounded Structured Sparsity')

parser.add_argument('--skip-pt', action='store_true', default=False, help='skip pretrain and simply load weights directly')
parser.add_argument('--path', type=str, default='', help='file to load pretrained weights from')
parser.add_argument('--model', type=str, default='vgg16', help='model to use, options: [vgg16, resnet50, inception_v3, alexnet]')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset to train on: [CIFAR10, ImageNet]')
parser.add_argument("--data-dir", type=str, default='/root/hostPublic/ImageNet/', help="the path to dataset folder.")

parser.add_argument('--ckpt-dir', type=str, default='', help='checkpoint save/load directory, default=ckpt/<modelName><time>/')
parser.add_argument('--epochs', type=int, default=100, help='pretrain number of epochs, default=100')
parser.add_argument('--batch', type=int, default=128, help='pretrain and finetune batch size, default=128')
parser.add_argument('--lr', type=float, default=0.01, help='pretrain initial learning rate, default=0.01')
parser.add_argument('--reg', type=float, default=5e-4, help='pretrain reg strength, default=5e-4')
parser.add_argument('--scheduler', type=str, default='cosine', help="scheduler ['step', 'cosine','coswm']")
parser.add_argument('--spar-reg', type=str, default='v2', help='sparsity regularizer type, options: [None, v1, v2, SSL]')
parser.add_argument('--spar-str', type=float, default=1e-4, help='sparsity reg strength, default=1e-4')
parser.add_argument('--scratch', action='store_false', default=True, help='train from scratch, default=False[ImageNet]')

parser.add_argument('--prune', action='store_true', default=False, help='apply prune, then finetune -- WARNING: must be a separate process from the initial pre-/re-training stage!')
parser.add_argument('--prune-type', type=str, default='cascade', help='pruning scheme, options: [percentage, std, dil, asym_dil, sintf, chunk, cascade, SSL]')
parser.add_argument('--q', type=float, default=0, help='prune threshold, will default to prune-type\'s default if not specified')

parser.add_argument('--ckpt-dir-ft', type=str, default='', help='finetune checkpoint directory, default=ckpt/finetune_<modelName><time>/')
parser.add_argument('--epochs-ft', type=int, default=50, help='finetune number of epochs, default=50')
parser.add_argument('--lr-ft', type=float, default=0.001, help='finetune initial learning rate, default=0.001')
parser.add_argument('--reg-ft', type=float, default=5e-6, help='finetune reg strength, default=5e-6')

parser.add_argument("--local_rank", type=int, default=-1)

args = parser.parse_args()

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

if args.model == "vgg16":
    if args.dataset == "ImageNet":
        model = vgg_in.vgg16(pretrained=args.scratch)
    else:
        model = vgg16.VGG16()
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
else:
    print("Model {} not supported!".format(args.model))
    sys.exit(0)
model = model.to(device)

#for layer in model.named_modules():
#    print(layer)
    
def replace_with_pruned(m, name):    
    #print(m)
    print("{}, {}".format(name, str(type(m))))
    if type(m) == PrunedConv or type(m) == PrunedLinear:
        return

    # HACK: directly replace conv layers of downsamples
    if name == "downsample":
        m[0] = PrunedConv(m[0])
    
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


if args.model != "vgg16" and args.model != "alexnet":
    replace_with_pruned(model, "model")
else:
    for i in range(len(model.features)):
        print(model.features[i])
        if isinstance(model.features[i], torch.nn.Conv2d):
            print("Replaced CONV")
            model.features[i] = PrunedConv(model.features[i])
    #for i in range(len(model.classifier)):
    #    if isinstance(model.classifier[i], torch.nn.Linear):
    #        print("Replaced Linear")
    #        model.classifier[i] = PrunedLinear(model.classifier[i])

if local_rank == 0:
        
    for layer in model.named_modules():
        print(layer)
print("WARNING: only CONV layers are targetted for pruning")

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
    elif args.dataset == "ImageNet":
        utils.train_imagenet(model, epochs=args.epochs, batch_size=eff_bs, lr=eff_lr, reg=args.reg, device=device,
                             checkpoint_path = args.ckpt_dir, spar_reg = args.spar_reg, spar_param = args.spar_str,
                             scheduler=args.scheduler, data_dir=args.data_dir, finetune=False, amp=True, lmdb=True)
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

if not args.prune:
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
                         scheduler=args.scheduler, data_dir=args.data_dir, finetune=True, amp=True, lmdb=True)
else:
    print("Dataset {} not suported!".format(args.dataset))
    sys.exit(0)

summary(model)
