import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist
from models import criterionsWT
from models.criterions import *
from data.BraTS import BraTS
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from tensorboardX import SummaryWriter
from torch import nn
from models.UNETR import UNETR
from CPS import *
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
parser = argparse.ArgumentParser()
# Basic Information
parser.add_argument('--user', default='Wangangcheng', type=str)
parser.add_argument('--experiment', default='SwinUNETR', type=str)
# parser.add_argument('--date', default='2023-12-07', type=str)
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
parser.add_argument('--description',
                    default='SwinUNETR,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='./Datasets', type=str)
parser.add_argument('--train_dir', default='EADC', type=str)
parser.add_argument('--val_dir', default='EADC', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--train_file', default='train.txt', type=str)
parser.add_argument('--val_file', default='valid.txt', type=str)
parser.add_argument('--dataset', default='EADC', type=str)
parser.add_argument('--model_name', default='SwinUNETR', type=str)

# Training Information
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=2e-5, type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
parser.add_argument('--criterion', default='softmax_dice2', type=str)
parser.add_argument('--num_cls', default=1, type=int)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--end_epoch', default=1000, type=int)
parser.add_argument('--val_epoch', default=100, type=int)
parser.add_argument('--save_freq', default=500, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--load', default=True, type=bool)
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
#parser.add_argument('--lora_dim', default=8, type=int)
parser.add_argument('--cps_dim', default=8, type=int)
args = parser.parse_args()
def freeze_parameters(model, param_names_to_freeze):
    for name, param in model.named_parameters():
        if name in param_names_to_freeze:
            param.requires_grad = False

param_names_to_freeze = [
    "module.encoder1.layer.conv1.conv.weight",
    "module.encoder1.layer.conv2.conv.weight",
    "module.encoder1.layer.conv3.conv.weight",
    "module.encoder2.layer.conv1.conv.weight",
    "module.encoder2.layer.conv2.conv.weight",
    "module.encoder3.layer.conv1.conv.weight",
    "module.encoder3.layer.conv2.conv.weight",
    "module.encoder4.layer.conv1.conv.weight",
    "module.encoder4.layer.conv2.conv.weight",
    "module.encoder10.layer.conv1.conv.weight",
    "module.encoder10.layer.conv2.conv.weight"
]
def main_worker():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
    log_file = log_dir + '.txt'
    log_args(log_file)
    logging.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')
    logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = SwinUNETR(img_size=128, in_channels=1, out_channels=2, feature_size=48)
    parameter_groups = get_optimizer_grouped_parameters(model, args.weight_decay)
    optimizer = torch.optim.Adam(parameter_groups, lr=args.lr)



    criterionWT = getattr(criterionsWT, args.criterion)

    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint',
                                      args.experiment + args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


    resume = './checkpoint/SwinUNETR2023-11-04/model_epoch_last.pth'
    writer = SummaryWriter()

    if os.path.isfile(resume):
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    if args.cla_dim > 0:
        model = convert_linear_layer_to_cps(model, 'swinViT', args.cps_dim)
        only_optimize_cps_parameters(model.swinViT)
        freeze_parameters(model, param_names_to_freeze)

    model.cuda()

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    val_list = os.path.join(args.root, args.val_dir, args.val_file)
    val_root = os.path.join(args.root, args.val_dir)

    train_set = BraTS(train_list, train_root, args.mode)

    val_set = BraTS(val_list, val_root, args.mode)


    logging.info('Samples for train = {}'.format(len(train_set)))
    logging.info('Samples for val = {}'.format(len(val_set)))



    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,shuffle=True,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size,shuffle=True,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)

    start_time = time.time()

    torch.set_grad_enabled(True)

    for epoch in range(args.start_epoch, args.end_epoch):

        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        start_epoch = time.time()

        # train
        model.train()
        for i, data in enumerate(tqdm(train_loader)):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            x, target = data
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(x)

            loss, loss_0, loss_1 = criterionWT(output, target)
            reduce_loss = loss.item()
            reduce_loss_0 = loss_0.item()
            reduce_lossWT = loss_1.item()


            logging.info('Epoch: {}_Iter:{}  loss: {:.5f} |0:{:.4f}|1:{:.4f} ||'
                         .format(epoch, i, reduce_loss, reduce_loss_0, reduce_lossWT))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        end_epoch = time.time()

        # val
        model.eval()
        if epoch % args.val_epoch == 0:
            logging.info('Samples for val = {}'.format(len(val_set)))
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
                    x, target = data
                    x = x.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    output = model(x)
                    lossWT = Dice(output[:, 1, ...], (target > 0).float())
                    logging.info('Epoch: {}_Iter:{}  Dice: WT:{:.4f}||'
                                 .format(epoch, i, 1 - lossWT))
        end_epoch = time.time()


        if (epoch + 1) % int(args.save_freq) == 0 \
                or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                or (epoch + 1) % int(args.end_epoch - 3) == 0:
            file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)
            writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('loss:', reduce_loss, epoch)
            writer.add_scalar('lossWT:', reduce_lossWT, epoch)


        epoch_time_minute = (end_epoch - start_epoch) / 60
        remaining_time_hour = (args.end_epoch - epoch - 1) * epoch_time_minute / 60
        logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
        logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))
    writer.close()

    final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
    torch.save({
        'epoch': args.end_epoch,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    },
        final_name)
    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)


def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
