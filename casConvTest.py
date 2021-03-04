#This works under my own filer_decomposition repo
import torch
from CasConv import casConv2d
from models import utils
from models import vgg16

model = vgg16.VGG16()
model.load_state_dict(torch.load("./ckpt/baseline/VGG16_93.49.pt"))
model.cuda()
for i in range(len(model.features)):
    if isinstance(model.features[i], torch.nn.Conv2d):
        model.features[i] = casConv2d(model.features[i])

utils.eval_cifar10(model, 32)