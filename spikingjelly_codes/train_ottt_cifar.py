from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import PIL
import time
from pathlib import Path
from PIL.Image import Image
import math
import torch.distributed as dist
import collections
from spikingjelly.activation_based import neuron, layer, functional
from spikingjelly.activation_based.model import spiking_vggws_ottt as vggmodel
from torchtoolbox.transform import Cutout
from copy import deepcopy


parser = argparse.ArgumentParser(description='PyTorch SNN Training')
# Basic settings
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--path', default='./data', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-name', default='', type=str, help='nickname for this trial.')
# SNN settings
parser.add_argument('-T', default=6, type=int, help='simulating time-steps')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--warmup', default=0, type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--nesterov', action='store_true')
parser.add_argument('--weight-decay', '--wd', default=0., type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--optimizer', default='SGD', type=str, help='which optimizer')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
# Miscs
parser.add_argument('--manualSeed', default=2022, type=int, help='manual seed')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--pin_memory', action='store_true')
# OTTT setting
parser.add_argument('--online', action='store_true', help='online update for each time step')
parser.add_argument('--loss-lambda', default=0.05, type=float)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}



# Use CUDA
use_cuda = torch.cuda.is_available()
if not use_cuda:
    raise RuntimeError('Need to use gpu.')
device = 'cuda'
print(args.local_rank)
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
current_iter = 0

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
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


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    args.T_max = args.epochs
    if args.online:
        args.model = 'vggws_ottto_'
    else:
        args.model = 'vggws_ottta_'

    if not os.path.isdir(args.checkpoint + '/' + args.dataset):
        os.makedirs(args.checkpoint + '/' + args.dataset)

    args.nprocs = torch.cuda.device_count()
    if args.nprocs > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        cudnn.benchmark = True
        args.train_batch = int(args.train_batch / args.nprocs)
        args.test_batch = int(args.test_batch / args.nprocs)

    print('==> Preparing dataset %s' % args.dataset)
    assert args.dataset == 'cifar10'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        Cutout(),
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

    trainset = datasets.CIFAR10(root=args.path, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=args.path, train=False, download=False, transform=transform_test)

    if args.nprocs > 1:
        train_sampler = data.distributed.DistributedSampler(trainset)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_memory)
        test_sampler = data.distributed.DistributedSampler(testset)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, sampler=test_sampler, num_workers=args.workers, pin_memory=args.pin_memory)
    else:
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=args.pin_memory)


    # Model
    print("==> creating model")

    model = vggmodel.ottt_spiking_vggws(num_classes=10, spiking_neuron=neuron.OTTTLIFNode)

    if args.nprocs > 1:
        model.cuda(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        #criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    else:
        model.cuda()
        cudnn.benchmark = True
        #criterion = nn.CrossEntropyLoss()

    def f_loss_t(y_t, target_t):
        target_t_onehot = F.one_hot(target_t, 10).float()
        loss = ((1 - args.loss_lambda) * F.cross_entropy(y_t, target_t) + args.loss_lambda * F.mse_loss(y_t, target_t_onehot)) / args.T
        return loss

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))


    title = args.dataset + 'SNN_Conv'
    logger = Logger(os.path.join(args.checkpoint, args.dataset, args.model+args.name + '.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])



    # Train and val
    for epoch in range(start_epoch, args.epochs):
        if args.nprocs > 1:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        if args.local_rank == 0 or args.local_rank == -1:
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, f_loss_t, optimizer, args, warmup=args.warmup)
        test_loss, test_acc = test(testloader, model, f_loss_t, args)


        # append logger file
        if args.local_rank == 0 or args.local_rank == -1:
            logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

            ## save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint+'/'+args.dataset, filename=args.model+args.name)

    logger.close()

    print('Best acc:')
    print(best_acc)


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def train(trainloader, model, f_loss_t, optimizer, args, warmup=0):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    global current_iter

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if warmup != 0 and current_iter < warmup:
            adjust_warmup_lr(optimizer, current_iter, warmup)
            current_iter += 1

        inputs, targets = inputs.cuda(), targets.cuda()

        inputs_seq = inputs.unsqueeze(1).repeat(1, args.T, *[1 for _ in range(len(inputs.shape[1:]))])
        targets_seq = targets.unsqueeze(1).repeat(1, args.T, *[1 for _ in range(len(targets.shape[1:]))])
        batch_loss, outputs_all = functional.ottt_online_training(model, optimizer, inputs_seq, targets_seq, f_loss_t, args.online)

        #outputs_all = []
        #batch_loss = 0.

        #if not args.online:
        #    optimizer.zero_grad(set_to_none=True)

        ## calculate gradient at each time step
        #for tt in range(args.T):
        #    if args.online:
        #        optimizer.zero_grad(set_to_none=True)
        #    inputs = inputs.detach()
        #    outputs = model(inputs)
        #    loss = f_loss_t(outputs, targets)
        #    outputs_all.append(outputs.detach())

        #    loss.backward()
        #    batch_loss += loss.data

        #    if args.online:
        #        optimizer.step()

        #if not args.online:
        #    optimizer.step()

        #outputs_all = torch.stack(outputs_all, dim=1)

        mean_out = outputs_all.mean(1)
        prec1, _ = accuracy(mean_out.data, targets.data, topk=(1, 5))
        if args.nprocs > 1:
            prec1 = reduce_tensor(prec1)
            batch_loss = reduce_tensor(batch_loss)
        top1.update(prec1.item(), inputs.size(0))
        losses.update(batch_loss.item(), inputs.size(0))

        functional.reset_net(model)

    return (losses.avg, top1.avg)


@torch.no_grad()
def test(testloader, model, f_loss_t, args):
    global best_acc
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs_all=[]
        batch_loss = 0.
        for tt in range(args.T):
            # compute output
            outputs = model(inputs)
            loss = f_loss_t(outputs, targets)
            outputs_all.append(outputs.detach())
            batch_loss += loss.data

        outputs_all = torch.stack(outputs_all, dim=1)
        mean_out = outputs_all.mean(1)
        prec1, _ = accuracy(mean_out.data, targets.data, topk=(1, 5))
        if args.nprocs > 1:
            prec1 = reduce_tensor(prec1)
            batch_loss = reduce_tensor(batch_loss)
        top1.update(prec1.item(), inputs.size(0))
        losses.update(batch_loss.item(), inputs.size(0))
        functional.reset_net(model)

    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint'):
    filepath = os.path.join(checkpoint, filename+'.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, filename+'_best.pth'))


def adjust_learning_rate(optimizer, epoch):
    global state
    state['lr'] = 0.5 * args.lr * (1 + math.cos(epoch/args.T_max*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']


def adjust_warmup_lr(optimizer, citer, warmup):
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr'] * (citer + 1.) / warmup


if __name__ == '__main__':
    whole_time = time.time()
    main()
    print('Whole running time:', time.time() - whole_time)
