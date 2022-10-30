import os
import logging
import numpy as np
from argparse import ArgumentParser

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import rnn
from torch.nn import functional
from transformers import BertTokenizer

from ranker.dataloader import read_data, create_data_from_ranker, RankBertDataset
from models.Ranker import Bert
from utils.log_utils import RankerLogger
from utils import torch_utils


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if args.multi_gpu:
        logging.info('run on multi GPU')
        torch.distributed.init_process_group(backend='nccl')

    torch_utils.setup_seed(0)

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model, optimizer, epoch, best_metric = torch_utils.load(args.pretrained_model_path)
    model = model.cpu() if args.use_cpu else model.cuda()

    # if args.multi_gpu:
    #     model = DistributedDataParallel(model, find_unused_parameters=True)

    if args.multi_gpu:
        saved_model = model.module
    else:
        saved_model = model

    torch_utils.save(output_path, 'best.pth', saved_model, optimizer, epoch, best_metric)
    logging.info("complete training")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--pretrained_model_path', type=str,
                        default='../../runtime/ranker/bert/pretrain/best.pth')
    parser.add_argument('--output_path', type=str,
                        default='../../runtime/ranker/bert/temp')
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--multi_gpu', type=bool,
                        help='run with: -m torch.distributed.launch',
                        default=True)
    parser.add_argument('--local_rank', type=int,
                        default=0)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
