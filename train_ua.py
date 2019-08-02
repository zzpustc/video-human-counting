import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from data.ua import UATrainDataset
from config.config import config
from layer.sst import build_sst
from utils.augmentations import SSJAugmentation, collate_fn
from layer.sst_loss import SSTLoss
import time
import torchvision.utils as vutils
from utils.operation import show_batch_circle_image

str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot Tracker Train')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--basenet', default=config['base_net_folder'], help='pretrained base model')
parser.add_argument('--batch_size', default=config['batch_size'], type=int, help='Batch size for training')
parser.add_argument('--resume', default=config['resume'], type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=config['num_workers'], type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=config['iterations'], type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=config['start_iter'], type=int, help='Begin counting iterations starting from this value (used with resume)')
parser.add_argument('--cuda', default=config['cuda'], type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=config['learning_rate'], type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--tensorboard',default=True, type=str2bool, help='Use tensor board x for loss visualization')
parser.add_argument('--port', default=6006, type=int, help='set vidom port')
parser.add_argument('--send_images', type=str2bool, default=True, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default=config['save_folder'], help='Location to save checkpoint models')
parser.add_argument('--ua_image_root', default=config['ua_image_root'], help='image folder of this dataset')
parser.add_argument('--ua_detection_root', default=config['ua_detection_root'], help='converted detection folder of the dataset')
parser.add_argument('--ua_ignore_root', default=config['ua_ignore_root'], help='ignore folder of the dataset')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if 'save_images_folder' in config and not os.path.exists(config['save_images_folder']):
    os.mkdir(config['save_images_folder'])

sst_dim = config['sst_dim']
means = config['mean_pixel']
batch_size = args.batch_size
max_iter = args.iterations
weight_decay = args.weight_decay

if 'learning_rate_decay_by_epoch' in config:
    stepvalues = list((config['epoch_size'] * i for i in config['learning_rate_decay_by_epoch']))
    save_weights_iteration = config['save_weight_every_epoch_num'] * config['epoch_size']
else:
    stepvalues = (90000, 95000)
    save_weights_iteration = 5000

gamma = args.gamma
momentum = args.momentum


if args.tensorboard:
    from tensorboardX import SummaryWriter
    if not os.path.exists(config['log_folder']):
        os.mkdir(config['log_folder'])
    writer = SummaryWriter(log_dir=config['log_folder'])

sst_net = build_sst('train')
net = sst_net

if args.cuda:
    net = torch.nn.DataParallel(sst_net)
    cudnn.benchmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    sst_net.load_weights(args.resume)
else:
    vgg_weights = torch.load(args.basenet)

    print('Loading the base network...')
    sst_net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    sst_net.extras.apply(weights_init)
    sst_net.selector.apply(weights_init)
    sst_net.final_net.apply(weights_init)


optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

criterion = SSTLoss(args.cuda)

def train():
    net.train()
    current_lr = config['learning_rate']
    print('Loading Dataset...')

    dataset = UATrainDataset(args.ua_image_root,
                             args.ua_detection_root,
                             args.ua_ignore_root,
                             SSJAugmentation(
                                 sst_dim, means
                             ))

    epoch_size = len(dataset) // args.batch_size
    print('Training SST on', dataset.dataset_name)
    step_index = 0

    batch_iterator = None

    data_loader = data.DataLoader(dataset, batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  pin_memory=False)

    # adjust the learning rate
    print('adjust the learning rate')
    for iteration in range(args.start_iter):
        if iteration in stepvalues:
            step_index += 1
            current_lr = adjust_learning_rate(optimizer, args.gamma, step_index)

    # start training
    for iteration in range(args.start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
            all_epoch_loss = []

        if iteration in stepvalues:
            step_index += 1
            current_lr = adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        img_pre, img_next, boxes_pre, boxes_next, labels, valid_pre, valid_next=\
            next(batch_iterator)

        if args.cuda:
            img_pre = Variable(img_pre.cuda())
            img_next = Variable(img_next.cuda())
            boxes_pre = Variable(boxes_pre.cuda())
            boxes_next = Variable(boxes_next.cuda())
            valid_pre = Variable(valid_pre.cuda(), volatile=True)
            valid_next = Variable(valid_next.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)

        else:
            img_pre = Variable(img_pre)
            img_next = Variable(img_next)
            boxes_pre = Variable(boxes_pre)
            boxes_next = Variable(boxes_next)
            valid_pre = Variable(valid_pre)
            valid_next = Variable(valid_next)
            labels = Variable(labels, volatile=True)


        # forward
        t0 = time.time()
        out = net(img_pre, img_next, boxes_pre, boxes_next, valid_pre, valid_next)

        optimizer.zero_grad()
        loss_pre, loss_next, loss_similarity, loss, accuracy_pre, accuracy_next, accuracy, predict_indexes = criterion(out, labels, valid_pre, valid_next)

        loss.backward()
        optimizer.step()
        t1 = time.time()

        all_epoch_loss += [loss.data.cpu()]

        if iteration % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ', ' + repr(epoch_size) + ' || epoch: %.4f ' % (iteration/(float)(epoch_size)) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')

        if args.tensorboard:
            if len(all_epoch_loss) > 30:
                writer.add_scalar('data/epoch_loss', float(np.mean(all_epoch_loss)), iteration)
            writer.add_scalar('data/learning_rate', current_lr, iteration)

            writer.add_scalar('loss/loss', loss.data.cpu(), iteration)
            writer.add_scalar('loss/loss_pre', loss_pre.data.cpu(), iteration)
            writer.add_scalar('loss/loss_next', loss_next.data.cpu(), iteration)
            writer.add_scalar('loss/loss_similarity', loss_similarity.data.cpu(), iteration)

            writer.add_scalar('accuracy/accuracy', accuracy.data.cpu(), iteration)
            writer.add_scalar('accuracy/accuracy_pre', accuracy_pre.data.cpu(), iteration)
            writer.add_scalar('accuracy/accuracy_next', accuracy_next.data.cpu(), iteration)

            # add weights
            if iteration % 1000 == 0:
                for name, param in net.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), iteration)

            # add images
            if args.send_images and iteration % 1000 == 0:
                result_image = show_batch_circle_image(img_pre, img_next, boxes_pre, boxes_next, valid_pre, valid_next, predict_indexes, iteration)

                writer.add_image('WithLabel/ImageResult', vutils.make_grid(result_image, nrow=2, normalize=True, scale_each=True), iteration)

        if iteration % save_weights_iteration == 0:
            print('Saving state, iter:', iteration)
            torch.save(sst_net.state_dict(),
                       os.path.join(
                           args.save_folder,
                           'sst300_0712_' + repr(iteration) + '.pth'))

    torch.save(sst_net.state_dict(), args.save_folder + '' + args.version + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
