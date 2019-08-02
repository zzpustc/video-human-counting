from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import json
import numpy as np
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from reid import models, datasets
from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator 
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler, caffeSampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.optim.sgd_caffe import SGD_caffe
from reid_model import Finetune
from tracker import SSTTracker, TrackerConfig, Track

def get_data(name, split_id, data_dir,
             height, width, crop_height, crop_width, batch_size,
             caffe_sampler=False,
             workers=4):

    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root, split_id=split_id)
    train_set = dataset.trainval
    num_classes = dataset.num_trainval_ids 

    # transforms
    train_transformer = T.Compose([
        T.RectScale(height, width),
#        T.CenterCrop((crop_height, crop_width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.RGB_to_BGR(),
        T.NormalizeBy(255),
    ])
    test_transformer = T.Compose([
        T.RectScale(height, width),
#        T.CenterCrop((crop_height, crop_width)),
        T.ToTensor(),
        T.RGB_to_BGR(),
        T.NormalizeBy(255),
    ])

    # dataloaders
    sampler = caffeSampler(train_set, name, batch_size=batch_size, root=dataset.images_dir) if caffe_sampler else \
              RandomIdentitySampler(train_set, 10) #TODO
    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=sampler,
        pin_memory=True, drop_last=True)
    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Log args
    print(args)
    args_dict = vars(args)
    with open(osp.join(args.logs_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f)

    # Create data loaders
    dataset, num_classes, train_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height, \
                 args.width, args.crop_height, args.crop_width, args.batch_size, \
                 args.caffe_sampler, \
                 args.workers)

    # Create model
    net_base = SSTTracker()
    net_finetune = Finetune().cuda()
    net_finetune.init_weights()
    
    # Criterion
    criterion = TripletLoss(margin=args.margin).cuda()

    # Optimizer
    optimizer = SGD_caffe(net_finetune.parameters(),
                          lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr * (0.1 **(epoch // 20))
        for g in optimizer.param_groups:
            g['lr'] = lr
#        lr = args.lr if epoch <= 200 else \
#            args.lr * (0.2 ** ((epoch-200)//200 + 1))
#        for g in optimizer.param_groups:
#            g['lr'] = lr * g.get('lr_mult', 1)
    def parse_data(inputs):
        imgs, _, pids, _ = inputs
        inputs = Variable(imgs).cuda()
        targets = Variable(pids.cuda())
        return inputs, targets

    for epoch in range(750):
        adjust_lr(epoch)
        loss_epoch_train = 0.0
        net_finetune.train()
        for i, inputs in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            inputs, targets = parse_data(inputs)
            outputs = inputs
            for k in range(35):
                outputs = net_base.sst.vgg[k](outputs)
            outputs = net_finetune(outputs).squeeze()
            loss, prec, num_eff, num_all = criterion(outputs, targets)
            loss_epoch_train += loss.data
            loss.backward()
            optimizer.step()

        loss_epoch_train /= 41
        loss_epoch_valid = 0.0
        net_finetune.eval()
        for i, inputs in enumerate(tqdm(test_loader)):
            inputs, targets = parse_data(inputs)
            outputs = inputs
            for k in range(35):
                outputs = net_base.sst.vgg[k](outputs)
            outputs = net_finetune(outputs).squeeze()
            loss, prec, num_eff, num_all = criterion(outputs, targets)
            loss_epoch_valid += loss.data
        loss_epoch_valid /= 108
        print('epoch:%d__train_loss:%.4f__valid_loss:%.4f'%(epoch,loss_epoch_train.data,loss_epoch_valid.data))

        torch.save(net_finetune.cpu().state_dict(),
            "./weights/reid_finetune_{}.pth".format(epoch+1))
        net_finetune.cuda()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Triplet loss classification")

    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=180)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=160)
    parser.add_argument('--width', type=int, default=80)
    parser.add_argument('--crop-height', type=int, default=160)
    parser.add_argument('--crop-width', type=int, default=80)
    # model
    parser.add_argument('-a', '--arch', type=str, default='inception_v1_cpm_pretrained',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--use-relu', dest='use_relu', action='store_true', default=False)
    parser.add_argument('--dilation', type=int, default=2)
    # loss
    parser.add_argument('--margin', type=float, default=0.2,
                        help="margin of the triplet loss, default: 0.2")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=750)
    parser.add_argument('--caffe-sampler', dest='caffe_sampler', action='store_true', default=False)
    # paths
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
