import time

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim

import os

from pruned_layers import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(net, epochs=100, batch_size=128, lr=0.01, reg=5e-4,
          checkpoint_path = '', spar_reg = None, spar_param = 0.0):
    """
    Training a network
    :param net:
    :param epochs:
    :param batch_size:
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

    start_epoch, best_acc = _load_checkpoint(net, optimizer, checkpoint_path, scheduler)
    best_acc_path = ''

    if checkpoint_path=='':
        checkpoint_path = os.path.join(os.path.curdir, 'ckpt/'+ net.__class__.__name__
                                       + time.strftime('%m%d%H%M', time.localtime()))
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    
    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            # compute standard loss
            loss = criterion(outputs, targets)
            # compute sparse_regularization loss
            loss = loss.view(1)
            if spar_reg is not None:
                reg_loss = torch.zeros_like(loss).to('cuda')
                if spar_reg == 'v1':
                    for n, m in net.named_modules():
                        if isinstance(m, PrunedConv) or isinstance(m, PruneLinear):
                            reg_loss += m.compute_group_lasso_v1()
                if spar_reg == 'v2':
                    for n, m in net.named_modules():
                        if isinstance(m, PrunedConv) or isinstance(m, PruneLinear):
                            reg_loss += m.compute_group_lasso_v2()
                if spar_reg == 'SSL':
                    for n, m in net.named_modules():
                        if isinstance(m, PrunedConv) or isinstance(m, PruneLinear):
                            reg_loss += m.compute_SSL()
                            
                loss += reg_loss * spar_param
            
            loss.backward()

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

        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving Weight...")
            if os.path.exists(best_acc_path):
                os.remove(best_acc_path)
            best_acc_path = os.path.join(checkpoint_path, "retrain_weight_%d_%.2f.pt"%(epoch, best_acc))
            torch.save(net.state_dict(), best_acc_path)

        if (epoch+1) % 10 == 0:
            _save_checkpoint(net, optimizer, epoch, best_acc, checkpoint_path, scheduler)


def finetune_after_prune(net, epochs=100, batch_size=128, lr=0.01, reg=5e-4, checkpoint_path = ''):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    
    start_epoch, best_acc = _load_checkpoint(net, optimizer, checkpoint_path, scheduler)
    best_acc_path = ''
    
    if checkpoint_path=='':
        checkpoint_path = os.path.join(os.path.curdir, 'ckpt/finetune_'+ net.__class__.__name__
                                       + time.strftime('%m%d%H%M', time.localtime()))
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    
    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # before optimizer.step(), manipulate the gradient
            """
            Zero the gradients of the pruned variables.
            """
            for n, m in net.named_modules():
                if isinstance(m, PrunedConv):
                    m.conv.weight.grad = m.conv.weight.grad.float() * m.mask.float()
                if isinstance(m, PruneLinear):
                    m.linear.weight.grad = m.linear.weight.grad.float() * m.mask.float()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % 16 == 0:
                end = time.time()
                num_examples_per_second = 16 * batch_size / (end - start)
                #print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      #% (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()
        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving Weight...")
            if os.path.exists(best_acc_path):
                os.remove(best_acc_path)
            best_acc_path = os.path.join(checkpoint_path, "pruned_weight_%d_%.2f.pt"%(epoch, best_acc))
            torch.save(net.state_dict(), best_acc_path)

        if (epoch+1) % 10 == 0:
            _save_checkpoint(net, optimizer, epoch, best_acc, checkpoint_path, scheduler)

def test(net):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            print("PASS")
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx >= 19: ##
                break ##
    #num_val_steps = len(testloader)
    num_val_steps = 20 ##
    val_acc = correct / total
    print("Test Loss=%.4f, Test accuracy=%.4f" % (test_loss / (num_val_steps), val_acc))
    return val_acc

def _save_checkpoint(net, optimizer, cur_epoch, best_acc, save_root, scheduler=None):
    ckpt = {'weight':net.state_dict(),
            'optim': optimizer.state_dict(),
            'cur_epoch':cur_epoch,
            'best_acc':best_acc}
    if scheduler is not None:
        ckpt['scheduler_dict'] = scheduler.state_dict()
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    save_path = os.path.join(save_root, "checkpoint_%d.ckpt"%cur_epoch)
    torch.save(ckpt, save_path)
    print("\033[36mCheckpoint Saved @%d epochs to %s\033[0m"%(cur_epoch+1, save_path))

def _load_checkpoint(net, optimizer, ckpt_path, scheduler=None):
    if not os.path.exists(ckpt_path):
        print("\033[31mCannot find checkpoint folder!\033[0m")
        print("\033[33mTrain From scratch!\033[0m")
        return 0, 0     #Start Epoch, Best Acc
    ckpt_list = os.listdir(ckpt_path)
    last_epoch = -1
    for ckpt_name in ckpt_list:
        if "checkpoint_" in ckpt_name:
            ckpt_epoch = int(ckpt_name.split(".")[0].split('_')[1])
            if ckpt_epoch>last_epoch:
                last_epoch = ckpt_epoch
    if last_epoch == -1:
        print("\033[33mNo checkpoint found!")
        print("Train From scratch!\033[0m")
        return 0, 0
    ckpt_file = os.path.join(ckpt_path, "checkpoint_%d.ckpt"%last_epoch)
    ckpt = torch.load(ckpt_file)
    print("\033[36mStarting from %d epoch.\033[0m"%(ckpt['cur_epoch']))
    net.train()       #This is important for BN
    net.load_state_dict(ckpt['weight'])
    optimizer.load_state_dict(ckpt['optim'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_dict'])
        
    return ckpt['cur_epoch'], ckpt['best_acc']
