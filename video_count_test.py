import torch

from reid_model import Finetune
from tracker import SSTTracker, TrackerConfig, Track



# When new target enter, reid between new target and object pool
def reid():
    net_finetune = Finetune()
    net_finetune.load_dict()

def main(args):
    base_net = SSTTracker()





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
    parser.add_argument('--lr', type=float, default=0.01,
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