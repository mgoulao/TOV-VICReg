# inspired by: https://github.com/facebookresearch/moco-v3/blob/main/moco

import argparse
import math
import os
from pathlib import Path
import random
import shutil
import time
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as torchvision_models

import tov_vicreg.models.builder as tov_vicreg_builder
import tov_vicreg.models.optimizer as tov_vicreg_optimizer

from tov_vicreg.dataset.dqn_dataset import MultiDQNReplayDataset, get_DQN_Replay_loader
from tov_vicreg.models.logger import Logger


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_tiny', 'sgi_resnet', 'resnet'] + torchvision_model_names

parser = argparse.ArgumentParser(description='TOV-VICReg Pre-Training')
parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
parser.add_argument('--experiment_name', default=None, type=str, help='')
parser.add_argument('--save_only_final',  action='store_true', help='')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_tiny',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096)')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=84, type=int,
                    help='')
parser.add_argument('--mlp', default='1024-1024-1024', type=str,
                    help='hidden dimension in MLPs (default: 3 layers with 1024)')
# Loss
parser.add_argument("--sim-coeff", type=float, default=25.0,
                    help='Invariance regularization loss coefficient')
parser.add_argument("--std-coeff", type=float, default=25.0,
                    help='Variance regularization loss coefficient')
parser.add_argument("--temporal-coeff", type=float, default=1.0,
                    help='Variance regularization loss coefficient')
parser.add_argument("--cov-coeff", type=float, default=1.0,
                    help='Covariance regularization loss coefficient')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')
parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base).""")

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=2, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

parser.add_argument('--tmp_data_path', default='/path/to/tmp/train/', type=str,
        help='Please specify path to a directory for the tmp data.')
parser.add_argument("--dqn_games", nargs="+", default=["Breakout"])
parser.add_argument("--dqn_checkpoints", nargs="+", default=[1, 5])
parser.add_argument('--dqn_frames', type=int, default=3, help='Number of frames per observation')
parser.add_argument('--dqn_single_dataset_max_size', type=int, default=1000, help='Maximum size of a single dataset')


def main():
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    train(args)

def train(args):
    # create model
    print("=> creating model '{}'".format(args.arch))

    model = tov_vicreg_builder.TOVVICReg(args)

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    if args.gpu is not None:
        print("Using GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    if args.optimizer == 'lars':
        optimizer = tov_vicreg_optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler()

    logger = Logger(name=args.experiment_name, type="ssl_train", group="tov", args=vars(args))
    logger.plots["singular_values"] = []
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = None

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    normalize = transforms.ConvertImageDtype(torch.float) # DQN Replay uses 0-255 uint8 and the Transformer expects a float

    augmentation1 = [
        transforms.RandomResizedCrop(args.image_size, scale=(args.crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur((7, 7), sigma=(.1, .2))], p=1.0),
        transforms.RandomHorizontalFlip(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(args.image_size, scale=(args.crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur((7, 7), sigma=(.1, .2))], p=0.1),
        transforms.RandomSolarize(120, p=0.2),
        transforms.RandomHorizontalFlip(),
        normalize
    ]

    augmentation3 = [
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        normalize
    ]

    train_dataset = MultiDQNReplayDataset(
        Path(args.data),
        args.dqn_games,
        args.dqn_checkpoints,
        args.dqn_frames,
        args.dqn_single_dataset_max_size,
        TwoCropsTransform(transforms.Compose(augmentation1),
                                    transforms.Compose(augmentation2)),
        add_adjacent=True,
        adjacent_transform=transforms.Compose(augmentation3)
    )

    train_loader = get_DQN_Replay_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        sampler=None
    )

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_one_epoch(train_loader, model, optimizer, scaler, logger, epoch, args)

        if epoch + 1 == args.epochs:
            save_encoder(model.backbone.state_dict(), os.path.join(args.output_dir, 'final_encoder.pth'))
        elif not args.save_only_final:
            save_encoder(model.backbone.state_dict(), os.path.join(args.output_dir, f'encoder_{epoch}.pth'))

    logger.close()


def train_one_epoch(train_loader, model, optimizer, scaler, logger, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True) # o_t with augmentation 1
            images[1] = images[1].cuda(args.gpu, non_blocking=True) # o_t with augmentation 2
            images[2] = images[2].cuda(args.gpu, non_blocking=True) # o_t-1 with augmentation 3
            images[3] = images[3].cuda(args.gpu, non_blocking=True) # o_t+1 with augmentation 3

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], images[2], images[3])

        losses.update(loss.item(), images[0].size(0))

        log_info = {"loss": loss.item(), **model.log}

        logger.log_step(log_info)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_encoder(state, filename='final_encoder.pth'):
    torch.save(state, filename)

class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


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
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
