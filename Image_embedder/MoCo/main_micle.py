# Code based on https://github.com/facebookresearch/moco
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
# from torchvision.models import resnet18
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import moco.loader
import moco.builder

from load_data import load_data
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MoCo')
parser.add_argument('--data', type='str', default='ADNI')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--ngpus_per_node', default=1, type=int, help='ngpus per node.')
parser.add_argument('--print_freq', default=100, type=int, help='print frequency.')
parser.add_argument('--gpu', default=0, type=int, help='Gpu index.')
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

def main():
    args = parser.parse_args()
    assert args.n_views == 2
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu = -1


    train_images, train_labels, extract_images, extract_names = load_data(args)
    print("preparing images -- end")
    train_images_list = []
    for i in range(len(train_images)):
        a = [train_images[i], medical_aug(train_images[i])]
        train_images_list.append(a)
    del train_images
    train_data_testing = TensorData(train_images_list, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_data_testing, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)	

    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp).cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    args.resume = './checkpoint_0190.pth.tar'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)

        if (epoch+1)%10 ==0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='./micle_abide/checkpoint_{:04d}.pth.tar'.format(epoch))
            
    with torch.cuda.device(args.gpu_index):

        extract_names = np.array(extract_names)
        model.eval()
        kkk = 0
        for im in extract_images:
            if kkk == 0:
                im_tf = im.to(args.device)
                _, _, anchor_features = model(im_tf)
                feat = torch.mean(anchor_features, dim=0).unsqueeze(0).detach().cpu()
            else:
                im_tf = im.to(args.device)
                _, _, anchor_features = model(im_tf)
                feat_ = torch.mean(anchor_features, dim=0).unsqueeze(0).detach().cpu()
                feat = torch.cat((feat,feat_),dim=0)
                del feat_
            kkk+=1
    n = feat.detach().cpu().numpy()
    np.savetxt('./extracted_feature/train_feature.csv',n,delimiter=',')
    np.savetxt('./extracted_feature/train_id.csv',extract_names,delimiter=',')

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    micles = AverageMeter('Micle', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, micles,top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[:,0] = images[:,0].cuda(args.gpu, non_blocking=True)
            images[:,1] = images[:,1].cuda(args.gpu, non_blocking=True)
        output, target, q = model(im_q=images[:,0].cuda(), im_k=images[:,1].cuda())
        loss = criterion(output, target)
        _1, _2, features = model(images[:,0].cuda(), images[:,0].cuda())
        micle_loss = micle(args,features, _)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        micles.update(micle_loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_with_micle = loss + micle_loss
        # loss.backward()
        loss_with_micle.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        args.print_freq = 1
        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def medical_aug(img_t, size=96, s =1):
    img_t = torch.FloatTensor(img_t)
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.ToTensor()])
    trans = transforms.ToPILImage()
    img = trans(img_t)
    aug_img = data_transforms(img).detach().numpy()
    return aug_img

def micle(args, features, indx):
        index = indx.detach().numpy()
        unique, counts = np.unique(index, return_counts=True)
        count_dict = dict(zip(unique, counts))
        loss = torch.zeros(1).to(args.device)
        mse = nn.MSELoss().to(args.device)
        count = 0
        edge_count = 0
        len_dict = 0
        for key in count_dict:
            len_dict +=1
            if count_dict[key]>2:
                which = np.where(index == key)[0]
                mask = torch.tensor(which).to(args.device)

                features_ = features[mask]
                features_ = F.normalize(features_, dim=1)
                similarity_matrix = torch.matmul(features_, features_.T)
                similarity_matrix = F.normalize(similarity_matrix)
                mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(args.device)
                positive = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

                labels = torch.ones(positive.shape,dtype=torch.long).to(args.device)

                loss += mse(positive.to(torch.float32),labels.to(torch.float32))
                count +=1
        if not (count==0):
            loss =loss/count
        return loss

class AverageMeter(object):
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
class TensorData():
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.LongTensor(y_data)

        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len

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
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
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
            # print(torch.reshape(correct[:k],(-1,)).shape)
            correct_k = torch.reshape(correct[:k],(-1,)).float().sum(0, keepdim=True)
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()