###################################################################
#   Provide essential utilities for model training, evaluation and analysis
#   Part of the filter decomposing project
#
#   UNFINISHED RESEARCH CODE
#   DO NOT DISTRIBUTE
#
#   Author: Shiyu Li
#   Date:   12/02/2019
#
#   Changelog:
#   2020-01-22 Merge the decomposing utilities.
##################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torchvision.datasets.mnist import MNIST
import torch.distributed as dist
import torch.cuda.amp as amp

from decompose.decomConv import DecomposedConv2D
from models.op_count import profile
from models.resnet_s import LambdaLayer
from tensorboardX import SummaryWriter
from models.dataset_lmdb import ImageFolderLMDB
from pruned_layers import *

from models.autoaugment import ImageNetPolicy
from models.mixncut import Cutout

#import torch.profiler as profiler

import numpy as np
import tqdm
import datetime
import time
import os
import copy


def spar_reg_func(model, loss, spar_method=None, finetune=False):
    reg_loss = 0
    if (spar_method is not None) and (finetune == False):
        reg_loss = torch.zeros_like(loss).to('cuda')
        # Sparsity Regularization
        if spar_method == 'l1':
            for n, m in model.named_parameters():
                if "coef" in n:
                    reg_loss += torch.sum(torch.abs(m))
        elif spar_method == 'p2':
            for n, m in model.named_modules():
                if isinstance(m, DecomposedConv2D):
                    reg_loss += m.compute_p2_loss(bits=4)
        elif spar_method == 'sgl':
            for n, m in model.named_modules():
                if isinstance(m, DecomposedConv2D):
                    reg_loss += torch.sum(torch.abs(m.coefs))
                    reg_loss += m.compute_group_lasso()
        elif spar_method == 'v1':
            for n, m in model.named_modules():
                if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                    reg_loss += m.compute_group_lasso_v1()
        elif spar_method == 'v2':
            for n, m in model.named_modules():
                if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                    reg_loss += m.compute_group_lasso_v2()
        elif spar_method == 'None':
            reg_loss = 0.
        else:
            print("Sparsity regularizer {} not supported!".format(spar_method))
            exit(0)
    return reg_loss


def train_mnist(model, epochs, batch_size=256, lr=0.01, reg=5e-4, checkpoint_path='',
                spar_reg=None, spar_param=0.0,
                finetune=False, cross=False, cross_interval=5):
    print('==> Preparing data..')
    data_train = MNIST('./data/mnist',
                       download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor()]))
    data_test = MNIST('./data/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=1024, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=reg, nesterov=True)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1)

    best_acc_path = ''
    best_acc = 0.0

    if checkpoint_path == '':
        checkpoint_path = os.path.join(os.path.curdir, 'ckpt/' + model.__class__.__name__
                                       + time.strftime('%m%d%H%M', time.localtime()))
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    global_steps = 0
    start = time.time()
    for epoch in range(epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # Swap training basis or coefficient

        # Interleave training bases and coefficient
        if cross and (epoch+1) % cross_interval == 0:
            print('Swaping Bases and Coefficient Training...')
            for _, m in model.named_modules():
                if isinstance(m, DecomposedConv2D):
                    m.coefs.requires_grad = not m.coefs.requires_grad
                    m.basis.requires_grad = not m.basis.requires_grad

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss = loss.view(1)
            reg_loss = spar_reg_func(model, loss, spar_reg, finetune)
            loss += reg_loss * spar_param

            loss.backward()

            # if finetune:
            #    for n, m in model.named_parameters():
            #        if 'coef' in n:
            #            m.grad[m==0] = 0

            if finetune:
                # before optimizer.step(), manipulate the gradient
                """
                Zero the gradients of the pruned variables.
                """
                for n, m in model.named_modules():
                    if isinstance(m, PrunedConv):
                        m.conv.weight.grad[m.conv.weight == 0] = 0
                        #m.conv.weight.grad = m.conv.weight.grad.float() * m.mask.float()
                    if isinstance(m, PrunedLinear):
                        m.linear.weight.grad[m.linear.weight == 0] = 0
                        #m.linear.weight.grad = m.linear.weight.grad.float() * m.mask.float()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % 16 == 0:
                end = time.time()
                num_examples_per_second = 16 * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()
            scheduler.step()

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to("cuda"), targets.to("cuda")
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(test_loader)
        val_acc = correct / total

        print("Test Loss=%.4f, Test acc=%.4f" %
              (test_loss / (num_val_steps), val_acc))

        if spar_reg is not None:
            sparse = 0
            total = 0
            for n, m in model.named_parameters():
                if 'coef' in n:
                    sparse += np.prod(list(m[m <= 1e-7].shape))
                    total += np.prod(list(m.shape))
            print("Sparsity Level %.6f" % (sparse/total))
            if spar_reg == 'sgl':
                scores = []
                for n, m in model.named_modules():
                    if isinstance(m, DecomposedConv2D):
                        # print("OK")
                        scores.append(m.compute_chunk_score())
                print("Layer Scores:", scores)
                print("Total Avg Scores:", np.mean(scores))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving Weight...")
            if os.path.exists(best_acc_path):
                os.remove(best_acc_path)
            best_acc_path = os.path.join(
                checkpoint_path, "retrain_weight_%d_%.2f.pt" % (epoch, best_acc))
            torch.save(model.state_dict(), best_acc_path)

    return best_acc


def eval_mnist(model):
    print('==> Preparing data..')
    data_test = MNIST('./data/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=1024, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(test_loader)
    val_acc = correct / total
    print("Test Loss=%.4f, Test acc=%.4f" %
          (test_loss / (num_val_steps), val_acc))

    return val_acc


def train_cifar100(model, epochs=100, batch_size=128, lr=0.01, reg=5e-4,
                   checkpoint_path='', spar_reg=None, spar_param=0.0,
                   scheduler='step', finetune=False, cross=False, cross_interval=5):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=reg, nesterov=True)
    if scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    elif scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

    start_epoch, best_acc = _load_checkpoint(
        model, optimizer, checkpoint_path, scheduler)
    best_acc_path = ''

    if checkpoint_path == '':
        checkpoint_path = os.path.join(os.path.curdir, 'ckpt/' + model.__class__.__name__
                                       + time.strftime('%m%d%H%M', time.localtime()))
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    criterion = nn.CrossEntropyLoss()

    global_steps = 0
    start = time.time()
    for epoch in range(start_epoch, epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # Swap training basis or coefficient

        # Interleave training bases and coefficient
        if cross and (epoch+1) % cross_interval == 0:
            print('Swaping Bases and Coefficient Training...')
            for _, m in model.named_modules():
                if isinstance(m, DecomposedConv2D):
                    m.coefs.requires_grad = not m.coefs.requires_grad
                    m.basis.requires_grad = not m.basis.requires_grad

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss = loss.view(1)
            reg_loss = spar_reg_func(model, loss, spar_reg, finetune)
            loss += reg_loss * spar_param

            loss.backward()

            # if finetune:
            #    for n, m in model.named_parameters():
            #        if 'coef' in n:
            #            m.grad[m==0] = 0

            if finetune:
                # before optimizer.step(), manipulate the gradient
                """
                Zero the gradients of the pruned variables.
                """
                for n, m in model.named_modules():
                    if isinstance(m, PrunedConv):
                        m.conv.weight.grad[m.conv.weight == 0] = 0
                        #m.conv.weight.grad = m.conv.weight.grad.float() * m.mask.float()
                    if isinstance(m, PrunedLinear):
                        m.linear.weight.grad[m.linear.weight == 0] = 0
                        #m.linear.weight.grad = m.linear.weight.grad.float() * m.mask.float()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % 16 == 0:
                end = time.time()
                num_examples_per_second = 16 * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()

        if scheduler is not None:
            scheduler.step()

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to("cuda"), targets.to("cuda")
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total

        print("Test Loss=%.4f, Test acc=%.4f" %
              (test_loss / (num_val_steps), val_acc))

        if spar_reg is not None:
            sparse = 0
            total = 0
            for n, m in model.named_parameters():
                if 'coef' in n:
                    sparse += np.prod(list(m[m <= 1e-7].shape))
                    total += np.prod(list(m.shape))
            print("Sparsity Level %.6f" % (sparse/total))
            if spar_reg == 'sgl':
                scores = []
                for n, m in model.named_modules():
                    if isinstance(m, DecomposedConv2D):
                        # print("OK")
                        scores.append(m.compute_chunk_score())
                print("Layer Scores:", scores)
                print("Total Avg Scores:", np.mean(scores))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving Weight...")
            if os.path.exists(best_acc_path):
                os.remove(best_acc_path)
            best_acc_path = os.path.join(
                checkpoint_path, "retrain_weight_%d_%.2f.pt" % (epoch, best_acc))
            torch.save(model.state_dict(), best_acc_path)

        if (epoch+1) % 10 == 0:
            _save_checkpoint(model, optimizer, epoch, best_acc,
                             checkpoint_path, scheduler)

    return best_acc


def eval_cifar100(model, batch_size=128):
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("Test Loss=%.4f, Test acc=%.4f" %
          (test_loss / (num_val_steps), val_acc))

    return val_acc


def train_cifar10(model, epochs=100, batch_size=128, lr=0.01, reg=5e-4,
                  checkpoint_path='', spar_reg=None, spar_param=0.0,
                  scheduler='step', finetune=False, cross=False, cross_interval=5):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=reg, nesterov=True)
    if scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    elif scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

    start_epoch, best_acc = _load_checkpoint(
        model, optimizer, checkpoint_path, scheduler)
    best_acc_path = ''

    if checkpoint_path == '':
        checkpoint_path = os.path.join(os.path.curdir, 'ckpt/' + model.__class__.__name__
                                       + time.strftime('%m%d%H%M', time.localtime()))
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    criterion = nn.CrossEntropyLoss()

    global_steps = 0
    start = time.time()
    for epoch in range(start_epoch, epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # Swap training basis or coefficient

        # Interleave training bases and coefficient
        if cross and (epoch+1) % cross_interval == 0:
            print('Swaping Bases and Coefficient Training...')
            for _, m in model.named_modules():
                if isinstance(m, DecomposedConv2D):
                    m.coefs.requires_grad = not m.coefs.requires_grad
                    m.basis.requires_grad = not m.basis.requires_grad

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss = loss.view(1)
            reg_loss = spar_reg_func(model, loss, spar_reg, finetune)
            loss += reg_loss * spar_param

            loss.backward()

            # if finetune:
            #    for n, m in model.named_parameters():
            #        if 'coef' in n:
            #            m.grad[m==0] = 0

            if finetune:
                # before optimizer.step(), manipulate the gradient
                """
                Zero the gradients of the pruned variables.
                """
                for n, m in model.named_modules():
                    if isinstance(m, PrunedConv):
                        m.conv.weight.grad[m.conv.weight == 0] = 0
                        #m.conv.weight.grad = m.conv.weight.grad.float() * m.mask.float()
                    if isinstance(m, PrunedLinear):
                        m.linear.weight.grad[m.linear.weight == 0] = 0
                        #m.linear.weight.grad = m.linear.weight.grad.float() * m.mask.float()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % 16 == 0:
                end = time.time()
                num_examples_per_second = 16 * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()

        if scheduler is not None:
            scheduler.step()

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to("cuda"), targets.to("cuda")
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total

        print("Test Loss=%.4f, Test acc=%.4f" %
              (test_loss / (num_val_steps), val_acc))

        """
        if spar_reg is not None:
            sparse = 0
            total = 0
            for n, m in model.named_parameters():
                if 'coef' in n:
                    sparse += np.prod(list(m[m<=1e-7].shape))
                    total += np.prod(list(m.shape))
            print("Sparsity Level %.6f"%(sparse/total))
            if spar_reg == 'sgl':
                scores = []
                for n, m in model.named_modules():
                    if isinstance(m, DecomposedConv2D):
                       #print("OK")
                       scores.append(m.compute_chunk_score_post(chunk_size=10, tol=1e-6))
                print("Layer Scores:", scores)
                print("Total Avg Scores:", np.mean(np.array(scores)[:, 0]))
        """

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving Weight...")
            if os.path.exists(best_acc_path):
                os.remove(best_acc_path)
            best_acc_path = os.path.join(
                checkpoint_path, "retrain_weight_%d_%.2f.pt" % (epoch, best_acc))
            torch.save(model.state_dict(), best_acc_path)

        if (epoch+1) % 10 == 0:
            _save_checkpoint(model, optimizer, epoch, best_acc,
                             checkpoint_path, scheduler)

    return best_acc


def eval_cifar10(model, batch_size=128):
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("Test Loss=%.4f, Test acc=%.4f" %
          (test_loss / (num_val_steps), val_acc))

    return val_acc


def train_imagenet(model, epochs=100, batch_size=128, lr=0.01, reg=5e-4,
                   checkpoint_path='', spar_reg=None, spar_param=0.0, device='cuda',
                   scheduler=None, finetune=False, cross=False, cross_interval=5,
                   data_dir="/root/hostPublic/ImageNet/", amp=False, lmdb=False):
    # data_dir="../../ILSVRC/Data/CLS-LOC/", amp=False, lmdb=False
    # ):

    local_rank = torch.distributed.get_rank()

    torch.backends.cudnn.benchmark = True
    if amp:
        model.ena_amp = True
    # DistributedDataParallel will divide and allocate batch_size to all
    # available GPUs if device_ids are not set
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank)

    print("Loading Data...")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if lmdb:
        traindir = os.path.join(data_dir, 'train.lmdb')
        valdir = os.path.join(data_dir, 'val.lmdb')
        val_dataset = ImageFolderLMDB(valdir,
                                      transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          # normalize, ## MAY NEED TO REMOVE
                                      ]))

        train_trans = [transforms.RandomResizedCrop(224),
                       transforms.RandomHorizontalFlip()]
        if not finetune:
            # Disable the augmentation for finetuning
            train_trans += [ImageNetPolicy(), Cutout()]

        train_trans.append(transforms.ToTensor())
        # normalize, ## MAY NEED TO REMOVE)
        train_dataset = ImageFolderLMDB(traindir,
                                        transforms.Compose(train_trans))
    else:
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        val_dataset = datasets.ImageFolder(valdir,
                                           transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               # normalize, ## MAY NEED TO REMOVE
                                           ]))

        train_dataset = datasets.ImageFolder(traindir,
                                             transforms.Compose([
                                                 transforms.RandomResizedCrop(
                                                     224),
                                                 transforms.RandomHorizontalFlip(),
                                                 ImageNetPolicy(),
                                                 transforms.ToTensor(),
                                                 # normalize, ## MAY NEED TO REMOVE
                                             ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    testloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, sampler=val_sampler,
        num_workers=2, pin_memory=True)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
        num_workers=6, pin_memory=True)

    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=reg)

    if scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    elif scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
    elif scheduler == 'coswm':  # Cosine Scheduler with warmup
        def warm_up_with_cosine_lr(epoch): return (epoch + 1) / 5 if epoch < 5 \
            else 0.5 * (math.cos((epoch - 5) / (epochs - 5) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warm_up_with_cosine_lr)

    if checkpoint_path != '' and not finetune:
        _load_checkpoint(model, optimizer, checkpoint_path, scheduler)

    _train(model, trainloader, testloader, optimizer, epochs,
           scheduler, checkpoint_path, finetune=finetune, device=device,
           cross=cross, cross_interval=cross_interval,
           spar_method=spar_reg, spar_reg=spar_param,
           sampler=train_sampler, ena_amp=amp)


def val_imagenet(model, valdir="/root/hostPublic/ImageNet/", lmdb=False, amp=False, device='cuda'):
    torch.backends.cudnn.benchmark = True
    local_rank = torch.distributed.get_rank()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    print("Loading Validation Data...")

    if lmdb:
        valdir = os.path.join(valdir, 'val.lmdb')
        val_dataset = ImageFolderLMDB(valdir,
                                      transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          # normalize, ## MAY NEED TO REMOVE
                                      ]))
    else:
        valdir = os.path.join(valdir, 'val')
        val_dataset = datasets.ImageFolder(valdir,
                                           transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               # normalize, ## MAY NEED TO REMOVE
                                           ]))

    # val_loader = torch.utils.data.DataLoader(
    #    val_dataset,batch_size=256, shuffle=False, num_workers= 16, pin_memory=True)

    if amp:
        model.ena_amp = True

    # DistributedDataParallel will divide and allocate batch_size to all
    # available GPUs if device_ids are not set
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, sampler=val_sampler,
        num_workers=8, pin_memory=True)

    return _eval(model, val_loader, device)

# Fix for 1x1 Conv layer's SGL


def compute_sgl(m, chunk_size=32):
    layer_loss = torch.zeros(1).cuda()

    num_chunks = int((m.out_channels - 1) // chunk_size + 1)
    if num_chunks * chunk_size == m.out_channels:
        loss = torch.norm(
            torch.norm(m.weight.view(num_chunks, chunk_size, m.in_channels), p=1, dim=1), p=2)
    else:
        coef_view = m.weight.view(m.out_channels, m.in_channels)
        partial_l1 = torch.norm(coef_view[:(num_chunks - 1) * chunk_size, :].view(num_chunks - 1, chunk_size, -1), p=1,
                                dim=1)
        residual_l1 = torch.norm(
            coef_view[(num_chunks - 1) * chunk_size:, :].view(1, -1, m.in_channels), p=1, dim=1)
        loss = torch.norm(torch.cat((partial_l1, residual_l1), 0), p=2)
    return loss.cuda().view(1)


def _train(model, trainloader, testloader,  optimizer, epochs, scheduler=None,
           checkpoint_path='', save_interval=2, device='cuda', finetune=False,
           cross=False, cross_interval=5, spar_method=None, spar_reg=0.0, sampler=None, ena_amp=False):

    local_rank = torch.distributed.get_rank()

    if not finetune:
        start_epoch, best_acc = _load_checkpoint(
            model, optimizer, checkpoint_path, scheduler)
    else:
        start_epoch = 0
        best_acc = 0.0
    best_acc_path = ''

    if checkpoint_path == '':
        checkpoint_path = os.path.join(os.path.curdir, 'ckpt/' + model.__class__.__name__
                                       + time.strftime('%m%d%H%M', time.localtime()))
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    log_dir = os.path.join(checkpoint_path, 'log')
    writer = SummaryWriter(log_dir)

    criterion = nn.CrossEntropyLoss().to(device)
    if ena_amp:
        scaler = amp.GradScaler()

    train_coef = True
    for _, m in model.named_modules():
        if isinstance(m, DecomposedConv2D):
            m.coefs.requires_grad = True
            m.basis.requires_grad = True
    for n, m in model.named_modules():
        if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
            if finetune:
                m.finetune = True
            else:
                m.finetune = False

    end = time.time()
    for epoch in range(start_epoch, epochs):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(trainloader),
            [batch_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        model.train()
        print("Epoch: %d, Learning rate: ", optimizer.param_groups[0]['lr'])

        if sampler is not None:
            sampler.set_epoch(epoch)
        # Swap training basis or coefficient

        # if cross and (epoch+1)%cross_interval==0:      #Interleave training bases and coefficient
        #    train_coef = not train_coef
        #    print('Swaping Bases and Coefficient Training...')
        #    for _, m in model.named_modules():
        #        if isinstance(m, DecomposedConv2D):
        #            m.coefs.requires_grad = train_coef
        #            m.basis.requires_grad = not train_coef
        # elif not cross:
        #    for _, m in model.named_modules():
        #        if isinstance(m, DecomposedConv2D):
        #            m.coefs.requires_grad = True
        #            m.basis.requires_grad = True

        prefetcher = data_prefetcher(trainloader, device)
        inputs, targets = prefetcher.next()
        batch_idx = 0
        while inputs is not None:
            batch_idx += 1
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if ena_amp:
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss = loss.view(1)
                    reg_loss = torch.zeros_like(loss).to(device)

                    # Sparsity Regularization
                    if spar_method == 'l1':
                        for n, m in model.named_parameters():
                            if "coef" in n:
                                reg_loss += torch.sum(torch.abs(m))
                    elif spar_method == 'p2':
                        for n, m in model.named_modules():
                            if isinstance(m, DecomposedConv2D):
                                reg_loss += m.compute_p2_loss(bits=4)
                        # print("p2 loss is ", p2_loss.detach().cpu(), spar_param)
                    elif spar_method == 'sgl':
                        for n, m in model.named_modules():
                            if isinstance(m, DecomposedConv2D):
                                # print(m.compute_group_lasso().shape, reg_loss.shape, loss)
                                reg_loss += torch.sum(torch.abs(m.coefs))
                                reg_loss += m.compute_group_lasso(
                                    chunk_size=10)
                            elif isinstance(m, torch.nn.Conv2d) and m.weight.shape[2] == 0:
                                reg_loss += torch.sum(torch.abs(m.coefs))
                                reg_loss += compute_sgl(m, chunk_size=10)
                    elif spar_method == '4col':
                        for n, m in model.named_modules():
                            if isinstance(m, DecomposedConv2D):
                                # print(m.compute_4col_loss().detach().cpu())
                                reg_loss += m.compute_4col_loss()
                    elif spar_method == 'v1' and not finetune:
                        for n, m in model.named_modules():
                            if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                                reg_loss += m.compute_group_lasso_v1()
                    elif spar_method == 'v2' and not finetune:
                        # print("HERE")
                        for n, m in model.named_modules():
                            if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                                # m.compute_group_lasso_v2(device=device)
                                reg_loss += m.gl_loss
                    #print("Loss before reg: {}".format(loss))
                    #print("Loss of reg: {}".format(reg_loss * spar_reg))
                    loss += reg_loss * spar_reg
                scaler.scale(loss).backward()
                losses.update(scaler.scale(loss).item(), inputs.size(0))
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss.view(1)
                # Sparsity Regularization
                reg_loss = torch.zeros_like(loss).to(device)
                if spar_method == 'l1':
                    for n, m in model.named_parameters():
                        if "coef" in n:
                            reg_loss += torch.sum(torch.abs(m))
                    for n, m in model.named_modules():
                        # Prune 1x1 convolution
                        if isinstance(m, nn.Conv2d) and m.weight.shape[2] == 1:
                            reg_loss += torch.sum(torch.abs(m.weight))
                elif spar_method == 'naive_l1':
                    reg_loss = torch.zeros_like(loss).to(device)
                    for n, m in model.named_modules():
                        if isinstance(m, nn.Conv2d):
                            reg_loss += torch.sum(torch.abs(m.weight))
                elif spar_method == 'sgl':
                    for n, m in model.named_modules():
                        if isinstance(m, DecomposedConv2D):
                            reg_loss += torch.sum(torch.abs(m.coefs))
                            reg_loss += m.compute_group_lasso()
                        if isinstance(m, nn.Conv2d) and m.weight.shape[2] == 1:
                            reg_loss += torch.sum(torch.abs(m.weight))
                            reg_loss += compute_sgl(m)
                elif spar_method == 'v1' and not finetune:
                    for n, m in model.named_modules():
                        if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                            reg_loss += m.compute_group_lasso_v1()
                elif spar_method == 'v2' and not finetune:
                    for n, m in model.named_modules():
                        if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                            # m.compute_group_lasso_v2(device=device)
                            reg_loss += m.gl_loss
                loss += reg_loss * spar_reg
                loss.backward()
                losses.update(loss.item(), inputs.size(0))

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # if finetune:
            #    for n, m in model.named_parameters():
            #        if 'coef' in n:
            #            m.grad[m==0] = 0
            #    for n, m in model.named_modules():
            #        if isinstance(m, nn.Conv2d):
            #            m.weight.grad[m.weight==0] = 0

            if finetune:
                # before optimizer.step(), manipulate the gradient
                """
                Zero the gradients of the pruned variables.
                """
                for n, m in model.named_modules():
                    if isinstance(m, PrunedConv):
                        m.conv.weight.grad[m.conv.weight == 0] = 0
                        #m.conv.weight.grad = m.conv.weight.grad.float() * m.mask.float()
                    if isinstance(m, PrunedLinear):
                        m.linear.weight.grad[m.linear.weight == 0] = 0
                        #m.linear.weight.grad = m.linear.weight.grad.float() * m.mask.float()

            if ena_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            inputs, targets = prefetcher.next()
            # Prefetch and avoid sync
            # torch.cuda.synchronize()

            if batch_idx % 16 == 0:
                n_step = epoch * len(trainloader) + batch_idx
                if local_rank == 0:
                    progress.display(batch_idx)
                    if ena_amp:
                        writer.add_scalar(
                            'Train/Loss', scaler.scale(loss).item(), n_step)
                    else:
                        writer.add_scalar('Train/Loss', loss.item(), n_step)
                    writer.add_scalar('Train/Top1 ACC', top1.avg, n_step)
                    writer.add_scalar('Train/Top5 ACC', top5.avg, n_step)

            batch_time.update(time.time() - end)
            end = time.time()

        if scheduler is not None:
            scheduler.step()

        val_acc, top5_acc = _eval(model, testloader, device)
        val_acc = val_acc.item()
        top5_acc = top5_acc.item()
        if local_rank == 0:
            writer.add_scalar('Test/Top1 Acc', val_acc, epoch)
            writer.add_scalar('Test/Top5 Acc', top5_acc, epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                print("Saving Weight...")
                if os.path.exists(best_acc_path):
                    os.remove(best_acc_path)
                best_acc_path = os.path.join(
                    checkpoint_path, "retrain_weight_%d_%.2f.pt" % (epoch, best_acc))
                torch.save(model.state_dict(), best_acc_path)

            if (epoch+1) % save_interval == 0:
                _save_checkpoint(model, optimizer, epoch,
                                 best_acc, checkpoint_path, scheduler)

        else:
            if val_acc > best_acc:
                best_acc = val_acc

    return best_acc


def compute_chunk_score_post(m, chunk_size, tol=0.0):
    last_chunk = (m.out_channels) % chunk_size
    n_chunks = (m.out_channels) // chunk_size + (last_chunk != 0)

    coef_mat = m.weight.reshape((m.out_channels, -1))

    total_score = 0.
    total_nz_chunks = 0

    for chunk_idx in range(n_chunks):
        if chunk_idx == n_chunks - 1 and last_chunk != 0:
            current_chunk = coef_mat[chunk_idx * chunk_size:, :].detach()
        else:
            current_chunk = coef_mat[chunk_idx *
                                     chunk_size:(chunk_idx + 1) * chunk_size, :].detach()
        current_chunk[torch.abs(current_chunk) < tol] = 0.
        tmp_sum = torch.sum(torch.abs(current_chunk), dim=0)
        # print(tmp_sum)
        total_nz_chunks += torch.sum(tmp_sum != 0).cpu().item()
        local_score = torch.sum(tmp_sum == 0) / float(m.in_channels)
        total_score += local_score
    return total_score.cpu().item() / n_chunks, total_nz_chunks, n_chunks * m.in_channels


def _eval(model, testloader, device='cuda'):
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(testloader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    with torch.no_grad():
        prefetcher = data_prefetcher(testloader, device)
        inputs, targets = prefetcher.next()
        batch_idx = 0
        end = time.time()
        while inputs is not None:
            batch_idx += 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Gather Loss
            torch.distributed.all_reduce(loss, torch.distributed.ReduceOp.SUM)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            torch.distributed.all_reduce(acc1, torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(acc5, torch.distributed.ReduceOp.SUM)

            losses.update(loss.item()/world_size, targets.size(0))
            top1.update(acc1[0]/world_size, targets.size(0))
            top5.update(acc5[0]/world_size, targets.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            inputs, targets = prefetcher.next()
            if batch_idx % 10 == 0 and local_rank == 0:
                progress.display(batch_idx)
        if local_rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


class data_prefetcher():
    def __init__(self, loader, device=None):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225],
                                device=device).view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.to(
                self.device, non_blocking=True)
            self.next_target = self.next_target.to(
                self.device, non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def _save_checkpoint(model, optimizer, cur_epoch, best_acc, save_root, scheduler=None):
    ckpt = {'weight': model.state_dict(),
            'optim': optimizer.state_dict(),
            'cur_epoch': cur_epoch,
            'best_acc': best_acc}
    if scheduler is not None:
        ckpt['scheduler_dict'] = scheduler.state_dict()
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    save_path = os.path.join(save_root, "checkpoint_%d.ckpt" % cur_epoch)
    torch.save(ckpt, save_path)
    print("\033[36mCheckpoint Saved @%d epochs to %s\033[0m" %
          (cur_epoch+1, save_path))


def _load_checkpoint(model, optimizer, ckpt_path, scheduler=None):
    if not os.path.exists(ckpt_path):
        print("\033[31mCannot find checkpoint folder!\033[0m")
        print("\033[33mTrain From scratch!\033[0m")
        return 0, 0  # Start Epoch, Best Acc
    ckpt_list = os.listdir(ckpt_path)
    last_epoch = -1
    for ckpt_name in ckpt_list:
        if "checkpoint_" in ckpt_name:
            ckpt_epoch = int(ckpt_name.split(".")[0].split('_')[1])
            if ckpt_epoch > last_epoch:
                last_epoch = ckpt_epoch
    if last_epoch == -1:
        print("\033[33mNo checkpoint found!")
        print("Train From scratch!\033[0m")
        return 0, 0
    ckpt_file = os.path.join(ckpt_path, "checkpoint_%d.ckpt" % last_epoch)
    ckpt = torch.load(ckpt_file)
    print("\033[36mStarting from %d epoch.\033[0m" % (ckpt['cur_epoch']))
    model.train()  # This is important for BN
    model.load_state_dict(ckpt['weight'])
    optimizer.load_state_dict(ckpt['optim'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_dict'])

    return ckpt['cur_epoch'], ckpt['best_acc']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_non_zero(model, tol=1e-3):
    layers = []
    params = model.state_dict()
    for key, item in params.items():
        if "coefs" in key:
            param = item.cpu().numpy()
            x = np.sum((param >= tol), axis=1)
            num_nz, count = np.unique(x, return_counts=True)
            res = {}
            for i in range(len(count)):
                res[num_nz[i]] = count[i]
            print(res)
            layers.append(res)
    return layers


def operate_params(model, obj, func):
    '''
    :param model: The model to be operate
    :param obj: The correspond parameters to operate
    :param func: The operate function, receive parameter tensor as the input
    :return: The modified model
    '''
    ret_list = []
    for n, m in model.named_parameters():
        if obj in n:
            m.data, ret = func(m)
            ret_list.append(ret)
    return ret_list


def compute_sparsity(model):
    n_zero = 0
    n_coefs = 0
    n_param = 0
    for n, m in model.named_parameters():
        if 'coef' in n:
            n_zero += np.prod(list(m[m == 0].shape))
            n_coefs += np.prod(list(m.shape))

    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            n_param += np.prod(list(m.weight.shape))
            n_zero += np.prod(list(m.weight[m.weight == 0].shape))
    n_param += n_coefs
    if n_coefs == 0:
        print("No coefficients found!")
        n_coefs = n_param
    print("# of zero parameters: ", n_zero)
    print("# of coefficients: ", n_coefs)
    print("# of total parameters: ", n_param)
    print("Coefficient Sparsity: ", n_zero/n_coefs)
    print("Parameter Sparsity: ", n_zero/n_param)
    return n_zero/n_param, n_zero/n_coefs


def prune_by_std(model, obj='coef', s=1.0):
    with torch.no_grad():
        for n, m in model.named_parameters():
            if obj in n:
                thresh = np.std(m.detach().cpu().numpy()) * s
                m[torch.abs(m) < thresh] = 0
        for n, m in model.named_modules():
            # Prune 1x1 convolution
            if isinstance(m, nn.Conv2d) and m.weight.shape[2] == 1:
                thresh = np.std(m.weight.detach().cpu().numpy()) * s
                m.weight[torch.abs(m.weight) < thresh] = 0


def show_basis_angle(params, visualize=False):
    '''
    :param params: Basis, 3d array in shape (layer, num_base, base_dim)
                    Or a list of 2d matrix in shape (num_base, base_dim)
    :return: Cosine Similarity of each layer's bases
    '''
    if isinstance(params, list):
        num_layers = len(params)
        num_basis = [x.shape[0] for x in params]
    else:
        num_layers = params.shape[0]
        num_basis = [params.shape[1]] * num_layers

    layer_sim = []
    for ldx in range(num_layers):
        layer_sim.append(np.zeros((num_basis[ldx], num_basis[ldx])))
        for i in range(5):
            for j in range(5):
                layer_sim[ldx][i, j] = np.dot(params[ldx, i, :].flatten(), params[ldx, j, :].flatten()) / \
                    np.sqrt(np.sum(params[ldx, i, :] ** 2)
                            * np.sum(params[ldx, j, :] ** 2))
        if visualize:
            plt.matshow(layer_sim[ldx].squeeze())
            plt.title("Layer %d" % ldx)

    if isinstance(params, np.ndarray):
        layer_sim = np.array(layer_sim)

    return layer_sim

# For Sequential Models like VGG16 and AlexNet


def shrink(model, iterative=True, Linear_req=True):
    model.eval()  # Avoid Problem for BN, just for quick test and need revise
    in_remain = []
    in_redun = []
    out_redun = []
    out_remain = []
    basis_remain = []
    cors_coef = []
    shortcut_path = [0]
    shortcut_padding = []

    for n, m in model.named_modules():
        # print(n)
        if isinstance(m, DecomposedConv2D):
            temp = m.coefs.data.detach().cpu().numpy().reshape(
                (m.out_channels, m.in_channels, m.num_basis))
            cors_coef.append(temp)
            in_redun.append(np.nonzero(
                np.sum(np.sum(temp != 0, axis=0), axis=1) == 0)[0])
            out_redun.append(np.nonzero(
                np.sum(np.sum(temp != 0, axis=1), axis=1) == 0)[0])
            in_remain.append(np.nonzero(
                np.sum(np.sum(temp != 0, axis=0), axis=1) != 0)[0])
            out_remain.append(np.nonzero(
                np.sum(np.sum(temp != 0, axis=1), axis=1) != 0)[0])

            basis_remain.append(np.nonzero(
                np.sum(np.sum(temp != 0, axis=0), axis=0) != 0)[0])
        if 'shortcut' in n:
            # Indicate the end point indices of the current shorcut path
            shortcut_path.append(len(in_redun)-1)
            if isinstance(m, LambdaLayer):
                shortcut_padding.append(len(in_redun)-1)

    # Get the intersection of the input and output channels
    for i in range(1, len(in_remain)):
        out_remain[i-1] = list(set(out_remain[i-1]
                                   ).intersection(set(in_remain[i])))
        in_remain[i] = list(
            set(in_remain[i]).intersection(set(out_remain[i-1])))
        out_redun[i-1] = list(set(out_redun[i-1]).union(set(in_redun[i])))
        in_redun[i] = list(set(in_redun[i]).union(set(out_redun[i-1])))

    # Adjust for Skip Connection
    if shortcut_path != [0]:
        for j in range(1, len(shortcut_path)):
            for i in range(1, len(shortcut_path)):
                start_point = shortcut_path[i-1]
                end_point = shortcut_path[i]

                if end_point in shortcut_padding:  # Avoid Padding Issue
                    continue

                # DEBUG
                #print("ShortCut: ", start_point, "-->", end_point)

                joint_remain = list(set(out_remain[end_point]).union(
                    set(out_remain[start_point])))

                if end_point < len(in_remain) - 1:  # Not the last Layer
                    in_remain[end_point+1] = joint_remain
                out_remain[end_point] = joint_remain
                out_remain[start_point] = joint_remain
                in_remain[start_point+1] = joint_remain

                joint_redun = list(set(out_redun[end_point]).intersection(
                    set(out_redun[start_point])))
                if end_point < len(in_redun) - 1:  # Not the last Layer
                    in_redun[end_point+1] = joint_redun
                out_redun[end_point] = joint_redun
                out_redun[start_point] = joint_redun
                in_redun[start_point+1] = joint_redun

    if iterative:
        modify = True

        while modify:
            modify = False
            for i in range(len(in_remain)):
                if i in shortcut_padding or i + 2 in shortcut_padding:
                    continue
                if len(in_redun[i]) != 0:
                    if np.sum(cors_coef[i][:, in_redun[i], :]) != 0:
                        cors_coef[i][:, in_redun[i], :] = 0
                        modify = True

                if len(out_redun[i]) != 0:
                    # print(out_redun[i])
                    if np.sum(cors_coef[i][out_redun[i], :, :]) != 0:
                        cors_coef[i][out_redun[i], :, :] = 0
                        modify = True
                if modify:
                    in_redun[i] = np.nonzero(
                        np.sum(np.sum(cors_coef[i] != 0, axis=0), axis=1) == 0)[0]
                    out_redun[i] = np.nonzero(
                        np.sum(np.sum(cors_coef[i] != 0, axis=1), axis=1) == 0)[0]
                    in_remain[i] = np.nonzero(
                        np.sum(np.sum(cors_coef[i] != 0, axis=0), axis=1) != 0)[0]
                    out_remain[i] = np.nonzero(
                        np.sum(np.sum(cors_coef[i] != 0, axis=1), axis=1) != 0)[0]

                    basis_remain[i] = np.nonzero(
                        np.sum(np.sum(cors_coef[i] != 0, axis=0), axis=0) != 0)[0]
                # print(modify)

            for i in range(1, len(in_remain)):
                out_remain[i-1] = list(set(out_remain[i-1]
                                           ).intersection(set(in_remain[i])))
                in_remain[i] = list(
                    set(in_remain[i]).intersection(set(out_remain[i-1])))
                out_redun[i-1] = list(set(out_redun[i-1]
                                          ).union(set(in_redun[i])))
                in_redun[i] = list(set(in_redun[i]).union(set(out_redun[i-1])))

            # Adjust for Skip Connection
            # Adjust for Skip Connection
            if shortcut_path != [0]:
                for j in range(1, len(shortcut_path)):
                    for i in range(1, len(shortcut_path)):
                        start_point = shortcut_path[i - 1]
                        end_point = shortcut_path[i]

                        if end_point in shortcut_padding:  # Avoid Padding Issue
                            continue

                        # DEBUG
                        # print("ShortCut: ", start_point, "-->", end_point)

                        joint_remain = list(set(out_remain[end_point]).union(
                            set(out_remain[start_point])))

                        if end_point < len(in_remain) - 1:  # Not the last Layer
                            in_remain[end_point + 1] = joint_remain
                        out_remain[end_point] = joint_remain
                        out_remain[start_point] = joint_remain
                        in_remain[start_point + 1] = joint_remain

                        joint_redun = list(set(out_redun[end_point]).intersection(
                            set(out_redun[start_point])))
                        if end_point < len(in_redun) - 1:  # Not the last Layer
                            in_redun[end_point + 1] = joint_redun
                        out_redun[end_point] = joint_redun
                        out_redun[start_point] = joint_redun
                        in_redun[start_point + 1] = joint_redun

    # Construct New Model
    new_model = copy.deepcopy(model)

    i = 0

    for n, m in new_model.named_modules():
        if isinstance(m, DecomposedConv2D):

            skip = False
            if i in shortcut_padding or i + 2 in shortcut_padding:
                print("Skip the border of the layer.")
                skip = True

            print("Shrinking The Layer:", n)

            ori_basis = m.basis.detach().cpu().numpy()
            ori_coefs = m.coefs.detach().cpu().numpy().reshape(
                (m.out_channels, m.in_channels, m.num_basis))

            #ori_bias = m.bias.detach().cpu().numpy()
            m.in_channels = len(in_remain[i])
            m.out_channels = m.out_channels if skip else len(out_remain[i])
            m.num_basis = len(basis_remain[i])

            print(i, " |----->New input Channel", m.in_channels,
                  " | New Output Channel", m.out_channels, " | New Basis", m.num_basis)

            r_basis_idx = basis_remain[i]
            r_in_channels_idx = in_remain[i]
            r_out_channels_idx = np.arange(
                m.out_channels) if skip else out_remain[i]

            new_basis = np.zeros(
                (m.num_basis, m.kernel_size[0]*m.kernel_size[1]), dtype=np.float32)
            new_coefs = np.zeros(
                (m.out_channels, m.in_channels, m.num_basis),  dtype=np.float32)
            #new_bias = ori_bias[r_out_channels_idx]

            for bidx in range(m.num_basis):
                new_basis[bidx, :] = ori_basis[r_basis_idx[bidx], :]

            for cin in range(m.in_channels):
                for cout in range(m.out_channels):
                    # print(ori_coefs.shape)
                    # print(new_coefs.shape)
                    new_coefs[cout, cin, :] = ori_coefs[r_out_channels_idx[cout],
                                                        r_in_channels_idx[cin], r_basis_idx]

            m.coefs = nn.Parameter(torch.tensor(new_coefs.reshape(
                m.out_channels*m.in_channels, m.num_basis)), requires_grad=True)
            m.basis = nn.Parameter(torch.tensor(
                new_basis), requires_grad=False)
            #m.bias = nn.Parameter(torch.tensor(new_bias), requires_grad=True)
            i += 1

        if isinstance(m, torch.nn.BatchNorm2d):
            print("Shrinking The BN Layer:", n)
            m.num_features = len(out_remain[i-1])
            print("----->New number of features: ", m.num_features)
            ori_weight = m.weight[list(out_remain[i-1])].detach().cpu().numpy()
            ori_bias = m.bias[list(out_remain[i-1])].detach().cpu().numpy()
            ori_mean = m.running_mean[list(
                out_remain[i-1])].detach().cpu().numpy()
            ori_var = m.running_var[list(
                out_remain[i-1])].detach().cpu().numpy()

            m.weight = nn.Parameter(torch.tensor(ori_weight))
            m.bias = nn.Parameter(torch.tensor(ori_bias))
            m.running_mean = torch.tensor(ori_mean)
            m.running_var = torch.tensor(ori_var)

        if isinstance(m, torch.nn.Linear) and Linear_req:
            print("Shrinking The First Linear Layer:", n)
            m.in_features = len(out_remain[i-1])
            ori_weight = m.weight.detach().cpu().numpy()
            new_weight = ori_weight[:, out_remain[i-1]]
            m.weight = nn.Parameter(torch.Tensor(new_weight))
            Linear_req = False

    return new_model, [len(x) for x in out_remain]


# Temperary implementation of shrinking function for ResNet50
# More redundancy could be explored if we use padding scheme or something else with non-downsampling connection
def shrink_resnet(model, iterative):
    model.eval()  # Avoid Problem for BN, just for quick test and need revise

    in_remain = []
    in_redun = []
    out_redun = []
    out_remain = []
    basis_remain = []
    cors_coef = []

    sc_in_remain = []
    sc_in_redun = []
    sc_out_redun = []
    sc_out_remain = []
    sc_cors_coef = []

    shortcut_path = [0]

    # Derive Redundancy
    for n, m in model.named_modules():
        # print(n)
        if isinstance(m, DecomposedConv2D) or (isinstance(m, nn.Conv2d) and m.weight.shape[2] == 1):
            if isinstance(m, DecomposedConv2D):
                temp = m.coefs.data.detach().cpu().numpy().reshape(
                    (m.out_channels, m.in_channels, m.num_basis))
            else:  # 1x1 Conv
                temp = m.weight.data.detach().cpu().numpy().reshape(
                    (m.out_channels, m.in_channels, 1))

            if 'downsample' in n:
                shortcut_path.append(len(out_remain) - 1)
                sc_cors_coef.append(temp)
                sc_in_redun.append(np.nonzero(
                    np.sum(np.sum(temp != 0, axis=0), axis=1) == 0)[0])
                sc_out_redun.append(np.nonzero(
                    np.sum(np.sum(temp != 0, axis=1), axis=1) == 0)[0])
                sc_in_remain.append(np.nonzero(
                    np.sum(np.sum(temp != 0, axis=0), axis=1) != 0)[0])
                sc_out_remain.append(np.nonzero(
                    np.sum(np.sum(temp != 0, axis=1), axis=1) != 0)[0])
            else:
                cors_coef.append(temp)
                # Specially tailored for BottleNeck Structure
                if ".conv1" in n:  # Do not modify the input of the bottleneck
                    in_redun.append([])
                    in_remain.append(np.arange(m.in_channels))
                else:
                    in_redun.append(np.nonzero(
                        np.sum(np.sum(temp != 0, axis=0), axis=1) == 0)[0])
                    in_remain.append(np.nonzero(
                        np.sum(np.sum(temp != 0, axis=0), axis=1) != 0)[0])

                if ".conv3" in n:
                    out_redun.append([])
                    out_remain.append(np.arange(m.out_channels,))
                else:
                    out_redun.append(np.nonzero(
                        np.sum(np.sum(temp != 0, axis=1), axis=1) == 0)[0])
                    out_remain.append(np.nonzero(
                        np.sum(np.sum(temp != 0, axis=1), axis=1) != 0)[0])

                basis_remain.append(np.nonzero(
                    np.sum(np.sum(temp != 0, axis=0), axis=0) != 0)[0])

    # Adjust for Skip Connection
    if shortcut_path != [0]:
        for i in range(1, len(shortcut_path)):
            start_point = shortcut_path[i - 1]
            end_point = shortcut_path[i]

            # DEBUG
            print("ShortCut: ", start_point, "-->", end_point)
            joint_remain_start = list(
                set(out_remain[start_point]).union(set(sc_in_remain[i-1])))
            out_remain[start_point] = joint_remain_start
            sc_in_remain[i-1] = joint_remain_start
            joint_redun_start = list(
                set(out_redun[start_point]).intersection(set(sc_in_redun[i-1])))
            out_redun[start_point] = joint_redun_start
            sc_in_redun[i-1] = joint_redun_start

            joint_remain_end = list(
                set(out_remain[end_point]).union(set(sc_out_remain[i-1])))
            out_remain[end_point] = joint_remain_end
            sc_out_remain[i-1] = joint_remain_end
            joint_redun_end = list(
                set(out_redun[end_point]).intersection(set(sc_out_redun[i-1])))
            out_redun[end_point] = joint_redun_end
            sc_out_redun[i-1] = joint_redun_end

    # Get the intersection of the input and output channels
    for i in range(1, len(in_remain)):
        out_remain[i - 1] = list(set(out_remain[i - 1]
                                     ).intersection(set(in_remain[i])))
        in_remain[i] = list(
            set(in_remain[i]).intersection(set(out_remain[i - 1])))
        out_redun[i - 1] = list(set(out_redun[i - 1]).union(set(in_redun[i])))
        in_redun[i] = list(set(in_redun[i]).union(set(out_redun[i - 1])))

    if shortcut_path != [0]:
        for i in range(1, len(shortcut_path)):
            start_point = shortcut_path[i - 1]
            end_point = shortcut_path[i]

            sc_in_remain[i - 1] = out_remain[start_point]
            sc_in_redun[i - 1] = out_redun[start_point]

            sc_out_remain[i - 1] = out_remain[end_point]
            sc_out_redun[i - 1] = out_redun[end_point]

    if iterative:
        modify = True
        while modify:
            modify = False
            for i in range(len(in_remain)):
                if len(in_redun[i]) != 0:
                    if np.sum(cors_coef[i][:, in_redun[i], :]) != 0:
                        cors_coef[i][:, in_redun[i], :] = 0
                        in_redun[i] = np.nonzero(
                            np.sum(np.sum(cors_coef[i] != 0, axis=0), axis=1) == 0)[0]
                        in_remain[i] = np.nonzero(
                            np.sum(np.sum(cors_coef[i] != 0, axis=0), axis=1) != 0)[0]
                        modify = True

                if len(out_redun[i]) != 0:
                    # print(out_redun[i])
                    if np.sum(cors_coef[i][out_redun[i], :, :]) != 0:
                        cors_coef[i][out_redun[i], :, :] = 0
                        out_redun[i] = np.nonzero(
                            np.sum(np.sum(cors_coef[i] != 0, axis=1), axis=1) == 0)[0]
                        out_remain[i] = np.nonzero(
                            np.sum(np.sum(cors_coef[i] != 0, axis=1), axis=1) != 0)[0]
                        modify = True

                if modify:
                    basis_remain[i] = np.nonzero(
                        np.sum(np.sum(cors_coef[i] != 0, axis=0), axis=0) != 0)[0]
                # print(modify)
            for i in range(len(sc_in_remain)):
                if len(sc_in_redun[i]) != 0:
                    if np.sum(sc_cors_coef[i][:, sc_in_redun[i], :]) != 0:
                        sc_cors_coef[i][:, sc_in_redun[i], :] = 0
                        modify = True

                if len(sc_out_redun[i]) != 0:
                    # print(out_redun[i])
                    if np.sum(sc_cors_coef[i][sc_out_redun[i], :, :]) != 0:
                        sc_cors_coef[i][sc_out_redun[i], :, :] = 0
                        modify = True

                if modify:
                    sc_in_redun[i] = (np.nonzero(
                        np.sum(np.sum(sc_cors_coef[i] != 0, axis=0), axis=1) == 0)[0])
                    sc_out_redun[i] = (np.nonzero(
                        np.sum(np.sum(sc_cors_coef[i] != 0, axis=1), axis=1) == 0)[0])
                    sc_in_remain[i] = (np.nonzero(
                        np.sum(np.sum(sc_cors_coef[i] != 0, axis=0), axis=1) != 0)[0])
                    sc_out_remain[i] = (np.nonzero(
                        np.sum(np.sum(sc_cors_coef[i] != 0, axis=1), axis=1) != 0)[0])

            # Adjust for Skip Connection
            if shortcut_path != [0]:
                for i in range(1, len(shortcut_path)):
                    start_point = shortcut_path[i - 1]
                    end_point = shortcut_path[i]

                    # DEBUG
                    # print("ShortCut: ", start_point, "-->", end_point)
                    joint_remain_start = list(
                        set(out_remain[start_point]).union(set(sc_in_remain[i - 1])))
                    out_remain[start_point] = joint_remain_start
                    sc_in_remain[i - 1] = joint_remain_start
                    joint_redun_start = list(
                        set(out_redun[start_point]).intersection(set(sc_in_redun[i - 1])))
                    out_redun[start_point] = joint_redun_start
                    sc_in_redun[i - 1] = joint_redun_start

                    joint_remain_end = list(
                        set(out_remain[end_point]).union(set(sc_out_remain[i - 1])))
                    out_remain[end_point] = joint_remain_end
                    sc_out_remain[i - 1] = joint_remain_end
                    joint_redun_end = list(
                        set(out_redun[end_point]).intersection(set(sc_out_redun[i - 1])))
                    out_redun[end_point] = joint_redun_end
                    sc_out_redun[i - 1] = joint_redun_end

            # Get the intersection of the input and output channels
            for i in range(1, len(in_remain)):
                out_remain[i - 1] = list(set(out_remain[i - 1]
                                             ).intersection(set(in_remain[i])))
                in_remain[i] = list(
                    set(in_remain[i]).intersection(set(out_remain[i - 1])))
                out_redun[i - 1] = list(set(out_redun[i - 1]
                                            ).union(set(in_redun[i])))
                in_redun[i] = list(
                    set(in_redun[i]).union(set(out_redun[i - 1])))

            if shortcut_path != [0]:
                for i in range(1, len(shortcut_path)):
                    start_point = shortcut_path[i - 1]
                    end_point = shortcut_path[i]

                    sc_in_remain[i - 1] = out_remain[start_point]
                    sc_in_redun[i - 1] = out_redun[start_point]

                    sc_out_remain[i - 1] = out_remain[end_point]
                    sc_out_redun[i - 1] = out_redun[end_point]

    # Construct New Model
    new_model = copy.deepcopy(model)

    new_width = []

    i = 0
    sc = 0
    Linear_req = True
    for n, m in new_model.named_modules():
        if isinstance(m, DecomposedConv2D) or (isinstance(m, nn.Conv2d) and m.weight.shape[2] == 1):
            # if isinstance(m, DecomposedConv2D):
            new_width.append(m.out_channels)

            if isinstance(m, nn.Conv2d):
                if 'downsample' in n:
                    print("Shrinking The Shortcut:", n)
                else:
                    print("Shrinking The Layer:", n)
                ori_weight = m.weight.detach().cpu().numpy().reshape(
                    (m.out_channels, m.in_channels, 1, 1))
                m.in_channels = len(
                    sc_in_remain[sc]) if 'downsample' in n else len(in_remain[i])
                m.out_channels = len(
                    sc_out_remain[sc]) if 'downsample' in n else len(out_remain[i])

                print("----->New input Channel", m.in_channels,
                      " | New Output Channel", m.out_channels)

                r_in_channels_idx = sc_in_remain[sc] if 'downsample' in n else in_remain[i]
                r_out_channels_idx = sc_out_remain[sc] if 'downsample' in n else out_remain[i]

                new_weight = np.zeros(
                    (m.out_channels, m.in_channels, 1, 1), dtype=np.float32)

                for cin in range(m.in_channels):
                    for cout in range(m.out_channels):
                        new_weight[cout, cin, :, :] = ori_weight[r_out_channels_idx[cout],
                                                                 r_in_channels_idx[cin], :, :]

                m.weight = nn.Parameter(torch.tensor(
                    new_weight), requires_grad=True)
                if 'downsample' in n:
                    sc += 1
                else:
                    i += 1

            else:
                print("Shrinking The Layer:", n)

                ori_basis = m.basis.detach().cpu().numpy()
                ori_coefs = m.coefs.detach().cpu().numpy().reshape(
                    (m.out_channels, m.in_channels, m.num_basis))
                m.in_channels = len(in_remain[i])
                m.out_channels = len(out_remain[i])
                m.num_basis = len(basis_remain[i])

                print("----->New input Channel", m.in_channels, " | New Output Channel", m.out_channels, " | New Basis",
                      m.num_basis)

                r_basis_idx = basis_remain[i]
                r_in_channels_idx = in_remain[i]
                r_out_channels_idx = out_remain[i]

                new_basis = np.zeros(
                    (m.num_basis, m.kernel_size[0] * m.kernel_size[1]), dtype=np.float32)
                new_coefs = np.zeros(
                    (m.out_channels, m.in_channels, m.num_basis), dtype=np.float32)

                for bidx in range(m.num_basis):
                    new_basis[bidx, :] = ori_basis[r_basis_idx[bidx], :]

                for cin in range(m.in_channels):
                    for cout in range(m.out_channels):
                        # print(ori_coefs.shape)
                        # print(new_coefs.shape)
                        new_coefs[cout, cin, :] = ori_coefs[r_out_channels_idx[cout],
                                                            r_in_channels_idx[cin], r_basis_idx]

                m.coefs = nn.Parameter(torch.tensor(new_coefs.reshape(m.out_channels * m.in_channels, m.num_basis)),
                                       requires_grad=True)
                m.basis = nn.Parameter(torch.tensor(
                    new_basis), requires_grad=False)
                i += 1

        if isinstance(m, torch.nn.BatchNorm2d):
            print("Shrinking The BN Layer:", n)
            m.num_features = len(out_remain[i - 1])
            print("----->New number of features: ", m.num_features)
            ori_weight = m.weight[list(
                out_remain[i - 1])].detach().cpu().numpy()
            ori_bias = m.bias[list(out_remain[i - 1])].detach().cpu().numpy()
            ori_mean = m.running_mean[list(
                out_remain[i - 1])].detach().cpu().numpy()
            ori_var = m.running_var[list(
                out_remain[i - 1])].detach().cpu().numpy()

            m.weight = nn.Parameter(torch.tensor(ori_weight))
            m.bias = nn.Parameter(torch.tensor(ori_bias))
            m.running_mean = torch.tensor(ori_mean)
            m.running_var = torch.tensor(ori_var)

        if isinstance(m, torch.nn.Linear) and Linear_req:
            print("Shrinking The First Linear Layer:", n)
            m.in_features = len(out_remain[i - 1])
            ori_weight = m.weight.detach().cpu().numpy()
            new_weight = ori_weight[:, out_remain[i - 1]]
            m.weight = nn.Parameter(torch.Tensor(new_weight))
            Linear_req = False

    return new_model, new_width
