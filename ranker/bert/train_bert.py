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


def get_data(args, data_path):
    if args.stage == 'pretrain':
        train_data_path = '%s/train_rank.txt' % data_path
        test_data_path = '%s/valid_rank.txt' % data_path
    elif args.stage == 'finetune':
        train_data_path = '%s/train_rank_finetune.txt' % data_path
        test_data_path = '%s/valid_rank_finetune.txt' % data_path

    train_data = read_data(train_data_path, debug=args.debug)
    test_data = read_data(test_data_path, debug=args.debug)

    if args.release:
        train_data.extend(test_data)

    return train_data, test_data


def get_dataset(args, epoch, tokenizer, train_data, test_data):
    if args.stage == 'pretrain':
        negative_batch_size = args.negative_batch_size
    elif args.stage == 'finetune':
        negative_batch_size = 0

    train_dataset = RankBertDataset(train_data, tokenizer, args.max_question_len, args.max_context_len,
                                    epoch=epoch, negative_batch_size=negative_batch_size, with_sep=False)
    test_dataset = RankBertDataset(test_data, tokenizer, args.max_question_len, args.max_context_len,
                                   epoch=epoch, negative_batch_size=negative_batch_size, with_sep=False)

    return train_dataset, test_dataset


def get_dataloader(args, train_dataset, test_dataset):
    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      sampler=DistributedSampler(train_dataset))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     sampler=DistributedSampler(test_dataset))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     shuffle=False)

    return train_dataloader, test_dataloader


def data_collate_fn(data):
    positive_tokens_list = []
    positive_tokens_type_id_list = []
    negative_tokens_list = []
    negative_tokens_type_id_list = []
    labels = []
    for idx in range(len(data)):
        positive_tokens = torch.LongTensor(np.array(data[idx][0][0]))
        positive_tokens_type_id = torch.LongTensor(np.array(data[idx][0][1]))

        negative_tokens = [torch.LongTensor(np.array(item[0])) for item in data[idx][1]]
        negative_tokens_type_id = [torch.LongTensor(np.array(item[1])) for item in data[idx][1]]

        label = torch.LongTensor([0])

        positive_tokens_list.append(positive_tokens)
        positive_tokens_type_id_list.append(positive_tokens_type_id)
        negative_tokens_list.append(negative_tokens)
        negative_tokens_type_id_list.append(negative_tokens_type_id)
        labels.append(label)

    return positive_tokens_list, positive_tokens_type_id_list, negative_tokens_list, negative_tokens_type_id_list, labels


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()

        positive_tokens, positive_tokens_type_id, negative_tokens, negative_tokens_type_id, labels = data

        positive_tokens = rnn.pad_sequence(positive_tokens, batch_first=True)
        positive_tokens_type_id = rnn.pad_sequence(positive_tokens_type_id, batch_first=True)
        positive_tokens_mask = (positive_tokens > 0).int()
        positive_tokens = positive_tokens.cpu() if args.use_cpu else positive_tokens.cuda()
        positive_tokens_type_id = positive_tokens_type_id.cpu() if args.use_cpu else positive_tokens_type_id.cuda()
        positive_tokens_mask = positive_tokens_mask.cpu() if args.use_cpu else positive_tokens_mask.cuda()

        negative_tokens_list = []
        negative_tokens_type_id_list = []
        [negative_tokens_list.extend(item) for item in negative_tokens]
        [negative_tokens_type_id_list.extend(item) for item in negative_tokens_type_id]
        negative_tokens_list = rnn.pad_sequence(negative_tokens_list, batch_first=True)
        negative_tokens_type_id_list = rnn.pad_sequence(negative_tokens_type_id_list, batch_first=True)
        negative_tokens_mask = (negative_tokens_list > 0).int()
        negative_tokens_list = negative_tokens_list.cpu() if args.use_cpu else negative_tokens_list.cuda()
        negative_tokens_type_id_list = negative_tokens_type_id_list.cpu() if args.use_cpu else negative_tokens_type_id_list.cuda()
        negative_tokens_mask = negative_tokens_mask.cpu() if args.use_cpu else negative_tokens_mask.cuda()

        labels = rnn.pad_sequence(labels, batch_first=True).squeeze(1)
        labels = labels.cpu() if args.use_cpu else labels.cuda()

        logits = model(positive_tokens, positive_tokens_type_id, positive_tokens_mask,
                       negative_tokens_list, negative_tokens_type_id_list, negative_tokens_mask)

        if logits is None:
            continue

        loss = functional.cross_entropy(logits, labels)
        loss = loss.mean()
        loss_sum += loss.item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    loss_sum = loss_sum / len(dataset)
    return loss_sum


def evaluate(args, dataloader, model):
    total_num = 0
    correct_num = 0

    model.eval()

    if args.multi_gpu:
        model = model.module

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            positive_tokens, positive_tokens_type_id, negative_tokens, negative_tokens_type_id, labels = data

            positive_tokens = rnn.pad_sequence(positive_tokens, batch_first=True)
            positive_tokens_type_id = rnn.pad_sequence(positive_tokens_type_id, batch_first=True)
            positive_tokens_mask = (positive_tokens > 0).int()
            positive_tokens = positive_tokens.cpu() if args.use_cpu else positive_tokens.cuda()
            positive_tokens_type_id = positive_tokens_type_id.cpu() if args.use_cpu else positive_tokens_type_id.cuda()
            positive_tokens_mask = positive_tokens_mask.cpu() if args.use_cpu else positive_tokens_mask.cuda()

            negative_tokens_list = []
            negative_tokens_type_id_list = []
            [negative_tokens_list.extend(item) for item in negative_tokens]
            [negative_tokens_type_id_list.extend(item) for item in negative_tokens_type_id]
            negative_tokens_list = rnn.pad_sequence(negative_tokens_list, batch_first=True)
            negative_tokens_type_id_list = rnn.pad_sequence(negative_tokens_type_id_list, batch_first=True)
            negative_tokens_mask = (negative_tokens_list > 0).int()
            negative_tokens_list = negative_tokens_list.cpu() if args.use_cpu else negative_tokens_list.cuda()
            negative_tokens_type_id_list = negative_tokens_type_id_list.cpu() if args.use_cpu else negative_tokens_type_id_list.cuda()
            negative_tokens_mask = negative_tokens_mask.cpu() if args.use_cpu else negative_tokens_mask.cuda()

            labels = rnn.pad_sequence(labels, batch_first=True).squeeze(1)
            labels = labels.cpu().numpy()

            logits = model(positive_tokens, positive_tokens_type_id, positive_tokens_mask,
                           negative_tokens_list, negative_tokens_type_id_list, negative_tokens_mask)
            preds = torch.argmax(logits, dim=1)
            preds = preds.cpu().numpy()

            total_num += len(preds)
            correct_num += (preds == labels).sum()

    accuracy = float(correct_num) / total_num if total_num > 0 else 0
    return accuracy


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if args.multi_gpu:
        logging.info('run on multi GPU')
        torch.distributed.init_process_group(backend='nccl')

    torch_utils.setup_seed(0)

    output_path = '%s/%s' % (args.output_path, args.stage)
    if args.bert_freeze:
        output_path += '_freeze'
    if args.release:
        output_path += '_release'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = RankerLogger(data_path=output_path, log_file='train.log')

    logging.info("loading embedding")
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info("loading data")
    train_data, test_data = get_data(args, args.data_path)

    best_metric = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info("loading pretrained model")
        if args.stage == 'pretrain':
            model, optimizer, epoch, best_metric = torch_utils.load(args.pretrained_model_path)
            model = model.cpu() if args.use_cpu else model.cuda()
        elif args.stage == 'finetune':
            model, _, _, _ = torch_utils.load(args.pretrained_model_path)
            model = model.cpu() if args.use_cpu else model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        logging.info("creating model")
        model = Bert('%s/config.json' % args.pretrained_bert_path,
                     '%s/pytorch_model.bin' % args.pretrained_bert_path,
                     args.bert_freeze)
        model = model.cpu() if args.use_cpu else model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.multi_gpu:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    logging.info("begin training")
    while epoch < args.epoch_size:
        epoch += 1

        if args.stage == 'finetune' and (epoch % args.rebuild_epoch_size == 0):
            logging.info('reloading dataset')
            create_data_from_ranker(args.data_path, output_path, tokenizer, model, args.negative_batch_size,
                                    max_question_len=args.max_question_len, max_context_len=args.max_context_len,
                                    batch_size=args.batch_size, use_cpu=args.use_cpu, debug=args.debug)
            train_data, test_data = get_data(args, output_path)

        train_dataset, test_dataset = get_dataset(args, epoch, tokenizer, train_data, test_data)
        train_dataloader, test_dataloader = get_dataloader(args, train_dataset, test_dataset)

        num_train_steps = int(len(train_dataset) / args.batch_size * args.epoch_size)
        num_warmup_steps = int(num_train_steps * args.lr_warmup_proportion)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_warmup_steps, gamma=args.lr_decay_gamma)

        train_loss = train(args, train_dataset, train_dataloader, model, optimizer, lr_scheduler)
        test_acc = evaluate(args, test_dataloader, model)

        logging.info('epoch[%s/%s], train loss: %s' % (epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], test accuracy: %s' % (epoch, args.epoch_size, test_acc))

        if args.multi_gpu:
            saved_model = model.module
        else:
            saved_model = model

        remark = ''
        if test_acc > best_metric:
            best_metric = test_acc
            remark = 'best'
            torch_utils.save(output_path, 'best.pth', saved_model, optimizer, epoch, best_metric)

        torch_utils.save(output_path, 'last.pth', saved_model, optimizer, epoch, best_metric)
        model_logger.write(epoch, train_loss, test_acc, remark)

    model_logger.draw_plot()
    logging.info("complete training")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='../../data/dataset/')
    parser.add_argument('--pretrained_bert_path', type=str,
                        default='../../data/bert/bert-base-chinese/')
    parser.add_argument('--pretrained_model_path', type=str,
                        default=None)
    parser.add_argument('--output_path', type=str,
                        default='../../runtime/ranker/bert')
    parser.add_argument('--stage', type=str, choices=['pretrain', 'finetune'],
                        default='pretrain')
    parser.add_argument('--bert_freeze', type=bool,
                        default=False)
    parser.add_argument('--release', type=bool,
                        default=False)
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--batch_size', type=int,
                        default=2)
    parser.add_argument('--negative_batch_size', type=int,
                        default=24)
    parser.add_argument('--epoch_size', type=int,
                        default=100)
    parser.add_argument('--rebuild_epoch_size', type=int,
                        default=10)
    parser.add_argument('--learning_rate', type=float,
                        default=1e-6)
    parser.add_argument('--lr_warmup_proportion', type=float,
                        default=0.1)
    parser.add_argument('--lr_decay_gamma', type=float,
                        default=0.9)
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
