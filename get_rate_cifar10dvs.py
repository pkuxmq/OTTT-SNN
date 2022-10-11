import datetime
import os
import time
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import sys
from torch.cuda import amp
from models import spiking_vgg
from modules import neuron, surrogate
import argparse
import torch.utils.data as data
import torchvision.transforms as transforms
from datasets.augmentation import ToPILImage, Resize, Padding, RandomCrop, ToTensor, Normalize
from datasets.cifar10_dvs import CIFAR10DVS
import math
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

_seed_ = 2022
import random
random.seed(_seed_)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

def main():

    parser = argparse.ArgumentParser(description='Classify DVS-CIFAR10')
    parser.add_argument('-T', default=10, type=int, help='simulating time-steps')
    parser.add_argument('-tau', default=2., type=float)
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default=None)

    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')

    parser.add_argument('-model', type=str, default='online_spiking_vgg11_ws')
    parser.add_argument('-drop_rate', type=float, default=0.1)
    parser.add_argument('-weight_decay', type=float, default=0.0)
    parser.add_argument('-cnf', type=str)
    parser.add_argument('-loss_lambda', type=float, default=0.001)

    parser.add_argument('-gpu-id', default='0', type=str, help='gpu id')

    args = parser.parse_args()
    #print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


    transform_train = transforms.Compose([
        ToPILImage(),
        Resize(48),
        Padding(4),
        RandomCrop(size=48, consistent=True),
        ToTensor(),
        Normalize((0.2728, 0.1295), (0.2225, 0.1290)),
    ])
    
    transform_test = transforms.Compose([
        ToPILImage(),
        Resize(48),
        ToTensor(),
        Normalize((0.2728, 0.1295), (0.2225, 0.1290)),
    ])
    num_classes = 10
    
    trainset = CIFAR10DVS(args.data_dir, train=True, use_frame=True, frames_num=args.T, split_by='number', normalization=None, transform=transform_train)
    train_data_loader = data.DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.j)
    
    testset = CIFAR10DVS(args.data_dir, train=False, use_frame=True, frames_num=args.T, split_by='number', normalization=None, transform=transform_test)
    test_data_loader = data.DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j)

    net = spiking_vgg.__dict__[args.model](single_step_neuron=neuron.OnlineLIFNode, tau=args.tau, surrogate_function=surrogate.Sigmoid(), track_rate=True, c_in=2, num_classes=num_classes, neuron_dropout=args.drop_rate, grad_with_rate=True, fc_hw=1, v_reset=None)
    #print(net)
    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.cuda()



    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #start_epoch = checkpoint['epoch'] + 1
        #max_test_acc = checkpoint['max_test_acc']

    criterion_mse = nn.MSELoss(reduce=True)

    for epoch in range(1):
        start_time = time.time()

        net.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        bar = Bar('Processing', max=len(test_data_loader))

        test_loss = 0
        test_acc = 0
        test_samples = 0
        batch_idx = 0
        spikes_all = None
        dims = None
        with torch.no_grad():
            for frame, label in test_data_loader:
                batch_idx += 1
                frame = frame.float().cuda()
                label = label.cuda()
                t_step = args.T
                total_loss = 0

                for t in range(t_step):
                    input_frame = frame[:, t]
                    if t == 0:
                        out_fr = net(input_frame, init=True, save_spike=True)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = net(input_frame, save_spike=True)
                        total_fr += out_fr.clone().detach()
                        #total_fr = total_fr * (1 - 1. / args.tau) + out_fr
                    if args.loss_lambda > 0.0:
                        label_one_hot = F.one_hot(label, num_classes).float()
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                    else:
                        loss = F.cross_entropy(out_fr, label) / t_step
                    total_loss += loss
                    spikes_batch = net.get_spike()
                    if spikes_all is None:
                        spikes_all = []
                        dims = []
                        for i in range(len(spikes_batch)):
                            spikes_all.append(torch.sum(torch.mean(spikes_batch[i], dim=1)).item())
                            dims.append(spikes_batch[i].shape[1])
                    else:
                        for i in range(len(spikes_all)):
                            spikes_all[i] = spikes_all[i] + torch.sum(torch.mean(spikes_batch[i], dim=1)).item()

                test_samples += label.numel()
                test_loss += total_loss.item() * label.numel()
                test_acc += (total_fr.argmax(1) == label).float().sum().item()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
                losses.update(total_loss, input_frame.size(0))
                top1.update(prec1.item(), input_frame.size(0))
                top5.update(prec5.item(), input_frame.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=batch_idx,
                            size=len(test_data_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                            )
                bar.next()
        bar.finish()

        test_loss /= test_samples
        test_acc /= test_samples
        for i in range(len(spikes_all)):
            spikes_all[i] = spikes_all[i] / (test_samples * t_step)
        total_rate = 0.
        total_dim = 0
        for i in range(len(spikes_all)):
            total_rate += spikes_all[i] * dims[i]
            total_dim += dims[i]
        total_rate /= total_dim

        total_time = time.time() - start_time

        print(f'test_loss={test_loss}, test_acc={test_acc}, total_time={total_time}')
        for i in range(len(spikes_all)):
            print(f'layer={i+1}, spike_rate={spikes_all[i]}')
        print(f'total_spike_rate={total_rate}')

if __name__ == '__main__':
    main()
