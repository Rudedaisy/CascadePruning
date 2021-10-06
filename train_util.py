import time

import torch
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import os

from tensorboardX import SummaryWriter

from pruned_layers import *

device = 'cuda' #if torch.cuda.is_available() else 'cpu'

def train(dataset, model, finetune=False, epochs=100, batch_size=128, lr=0.01, reg=5e-4, spar_reg=None, spar_param=0.0, checkpoint_path=''):
    if dataset == "CIFAR10":
        train_cifar10(model, finetune, epochs, batch_size, lr, reg, checkpoint_path, spar_reg, spar_param)
    elif dataset == "ImageNet":
        train_imagenet(model, finetune, epochs, batch_size, lr, reg, checkpoint_path, spar_reg, spar_param)
    else:
        assert False, "unsupported dataset specified!"

def train_cifar10(model, finetune=False, epochs=100, batch_size=128, lr=0.01, reg=5e-4,
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    
    _train("CIFAR10", model, trainloader, testloader, optimizer, epochs, batch_size,
           scheduler, checkpoint_path, finetune=finetune,
           spar_reg=spar_reg, spar_param=spar_param,
           sampler=None)

def train_imagenet(model, finetune=False, epochs=100, batch_size=128, lr=0.01, reg=5e-4,
                   checkpoint_path = '', spar_reg = None, spar_param = 0.0, data_dir="/root/hostPublic/ImageNet/"):
    if not torch.distributed.is_initialized():
        port = np.random.randint(10000, 65536)
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d'%port, rank=0, world_size=1)

    model.cuda()
    torch.backends.cudnn.benchmark = True

    # DistributedDataParallel will divide and allocate batch_size to all
    # available GPUs if device_ids are not set
    model = torch.nn.parallel.DistributedDataParallel(model)

    print("Loading Data...")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           #normalize,
                                       ]))
    
    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             #normalize,
                                         ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    testloader = torch.utils.data.DataLoader(
        val_dataset,batch_size=256, shuffle=False, sampler=val_sampler,
        num_workers=16, pin_memory=True)
    
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
        num_workers=32, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = reg, eps=1.0)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

    if checkpoint_path != '':
        _load_checkpoint(model, optimizer, checkpoint_path, scheduler)

    #if not os.path.exists(checkpoint_path):
    #    os.mkdir(checkpoint_path)
        
    _train("ImageNet", model, trainloader, testloader, optimizer, epochs,
           scheduler, checkpoint_path, finetune=finetune,
           spar_reg=spar_reg, spar_param=spar_param,
           sampler=train_sampler)
    
def test(dataset, model):
    if dataset == "CIFAR10":
        return test_cifar10(model)
    elif dataset == "ImageNet":
        return test_imagenet(model)
    else:
        assert False, "unsupported dataset!"
            
def test_cifar10(model):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("Test Loss=%.4f, Test accuracy=%.4f" % (test_loss / (num_val_steps), val_acc))
    return val_acc

def _save_checkpoint(model, optimizer, cur_epoch, best_acc, save_root, scheduler=None):
    ckpt = {'weight':model.state_dict(),
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

def _load_checkpoint(model, optimizer, ckpt_path, scheduler=None):
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
    model.train()       #This is important for BN
    model.load_state_dict(ckpt['weight'])
    optimizer.load_state_dict(ckpt['optim'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_dict'])
        
    return ckpt['cur_epoch'], ckpt['best_acc']

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

class data_prefetcher():
    def __init__(self, loader, dataset="ImageNet"):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        if dataset == "ImageNet":
            self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)
        elif dataset == "CIFAR10":
            self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).cuda().view(1,3,1,1)
            self.std = torch.tensor([0.2023, 0.1994, 0.2010]).cuda().view(1,3,1,1)
        else:
            assert False, "dataset not supported!"
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
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
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
    
def _train(dataset, model, trainloader, testloader,  optimizer, epochs, batch_size=100, scheduler=None,
           checkpoint_path='', save_interval=10, device='cuda', finetune=False,
           cross=False, cross_interval=5, spar_reg=None, spar_param = 0.0, sampler=None, ena_amp=False, ena_prefetcher=False):

    criterion = nn.CrossEntropyLoss().cuda()
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    best_acc_path = ''

    if finetune and checkpoint_path=='':
        checkpoint_path = os.path.join(os.path.curdir, 'ckpt/finetune_'+ model.__class__.__name__
                                       + time.strftime('%m%d%H%M', time.localtime()))
    elif not finetune and checkpoint_path=='':
            checkpoint_path = os.path.join(os.path.curdir, 'ckpt/'+ model.__class__.__name__
                                           + time.strftime('%m%d%H%M', time.localtime()))
    else:
        start_epoch, best_acc = _load_checkpoint(model, optimizer, checkpoint_path, scheduler)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    log_dir = os.path.join(checkpoint_path, 'log')
    writer = SummaryWriter(log_dir)

    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(trainloader),
            [batch_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        model.train()

        if sampler is not None:
            sampler.set_epoch(epoch)


        if dataset == "ImageNet":
            prefetcher = data_prefetcher(trainloader, dataset) 
            inputs, targets = prefetcher.next() 
        else:
            prefetcher = iter(trainloader)
            try:
                inputs, targets = next(prefetcher)
            except StopIteration:
                inputs = None
                targets = None
        batch_idx = 0
        
        train_loss = 0
        correct = 0
        total = 0
        #for batch_idx, (inputs, targets) in enumerate(trainloader):
        while inputs is not None:
            batch_idx += 1
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # compute standard loss
            loss = criterion(outputs, targets)
            # compute sparse_regularization loss
            loss = loss.view(1)
            if not finetune and spar_reg is not None:
                reg_loss = torch.zeros_like(loss).to('cuda')
                if spar_reg == 'v1':
                    for n, m in model.named_modules():
                        if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                            reg_loss += m.compute_group_lasso_v1()
                if spar_reg == 'v2':
                    for n, m in model.named_modules():
                        if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                            reg_loss += m.compute_group_lasso_v2()
                if spar_reg == 'SSL':
                    for n, m in model.named_modules():
                        if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
                            reg_loss += m.compute_SSL()

                #print("Loss before reg: {}".format(loss))
                #print("Loss of reg: {}".format(reg_loss * spar_param))
                loss += reg_loss * spar_param
                

            loss.backward()

            #acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            #top1.update(acc1[0], inputs.size(0))
            #top5.update(acc5[0], inputs.size(0))

            if finetune:
                # before optimizer.step(), manipulate the gradient
                """
                Zero the gradients of the pruned variables.
                """
                for n, m in model.named_modules():
                    if isinstance(m, PrunedConv):
                        m.conv.weight.grad = m.conv.weight.grad.float() * m.mask.float()
                    if isinstance(m, PrunedLinear):
                        m.linear.weight.grad = m.linear.weight.grad.float() * m.mask.float()
            
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1
            torch.cuda.synchronize()
            end = time.time()
            batch_time.update(time.time() - end)

            if global_steps % 16 == 0:
                num_examples_per_second = 16 * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()

            if dataset == "ImageNet":
                inputs, targets = prefetcher.next()
            else:
                try:
                    inputs, targets = next(prefetcher)
                except StopIteration:
                    inputs = None
                    targets = None

        if scheduler is not None:
            scheduler.step()

        """
        Start the testing code.
        """
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            if dataset == "ImageNet":
                prefetcher_test = data_prefetcher(testloader, dataset)	
                inputs, targets = prefetcher_test.next()	
            else:
                prefetcher_test = iter(testloader)
                try:
                    inputs, targets = next(prefetcher_test)
                except StopIteration:
                    inputs = None
                    targets = None
            batch_idx = 0
            #for batch_idx, (inputs, targets) in enumerate(testloader):
            while inputs is not None:
                batch_idx += 1
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                inputs, targets = prefetcher_test.next()
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))
        #acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        #losses.update(loss.item(), targets.size(0))
        #top1.update(acc1[0], targets.size(0))
        #top5.update(acc5[0], targets.size(0))
        
        #writer.add_scalar('Test/Top1 Acc', top1.avg, epoch)
        #writer.add_scalar('Test/Top5 Acc', top5.avg, epoch)
        
        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving Weight...")
            if os.path.exists(best_acc_path):
                os.remove(best_acc_path)
            best_acc_path = os.path.join(checkpoint_path, "retrain_weight_%d_%.2f.pt"%(epoch, best_acc))
            torch.save(model.state_dict(), best_acc_path)

        if (epoch+1) % save_interval == 0:
            _save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path, scheduler)
