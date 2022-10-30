import os
import logging
import numpy as np
from argparse import ArgumentParser

import torch
from torch import optim
from torch.nn.utils import rnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import BertTokenizer

from validator.dataloader import ValidateDataset
from validator import evaluator
from models.Classifier import Bert
from utils.log_utils import ValidatorLogger
from utils import torch_utils


def get_dataset(args, tokenizer):
    train_dataset = ValidateDataset('%s/train_validate.txt' % args.data_path, tokenizer,
                                    args.max_question_len, args.max_context_len, with_sep=False, debug=args.debug)
    test_dataset = ValidateDataset('%s/valid_validate.txt' % args.data_path, tokenizer,
                                   args.max_question_len, args.max_context_len, with_sep=False, debug=args.debug)

    if args.release:
        train_dataset.data.extend(test_dataset.data)

    return train_dataset, test_dataset


def data_collate_fn(data):
    data = np.array(data)

    labels = data[:, 0].tolist()
    labels = torch.LongTensor(np.array(labels))

    tokens = data[:, 1].tolist()
    tokens = [torch.LongTensor(np.array(item)) for item in tokens]

    tokens_type_id = data[:, 2].tolist()
    tokens_type_id = [torch.LongTensor(np.array(item)) for item in tokens_type_id]

    tokens = rnn.pad_sequence(tokens, batch_first=True)
    tokens_type_id = rnn.pad_sequence(tokens_type_id, batch_first=True)
    tokens_mask = (tokens > 0).int()

    return labels, tokens, tokens_type_id, tokens_mask


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        labels, tokens, tokens_type_id, tokens_mask = data

        labels = labels.cpu() if args.use_cpu else labels.cuda()
        tokens = tokens.cpu() if args.use_cpu else tokens.cuda()
        tokens_type_id = tokens_type_id.cpu() if args.use_cpu else tokens_type_id.cuda()
        tokens_mask = tokens_mask.cpu() if args.use_cpu else tokens_mask.cuda()

        logits = model(tokens, tokens_type_id, tokens_mask)
        loss = F.cross_entropy(logits, labels)
        loss = loss.mean()

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    loss_sum = loss_sum / len(dataset)
    return loss_sum


def evaluate(args, dataloader, model):
    pred_answers = []
    gold_answers = []

    model.eval()

    if args.multi_gpu:
        model = model.module

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            labels, tokens, tokens_type_id, tokens_mask = data

            tokens = tokens.cpu() if args.use_cpu else tokens.cuda()
            tokens_type_id = tokens_type_id.cpu() if args.use_cpu else tokens_type_id.cuda()
            tokens_mask = tokens_mask.cpu() if args.use_cpu else tokens_mask.cuda()

            logits = model(tokens, tokens_type_id, tokens_mask)
            logits = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            pred_answers.extend(preds)
            gold_answers.extend(labels)

    acc, pre, rec, f1 = evaluator.evaluate(gold_answers, pred_answers)
    return acc, pre, rec, f1


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if args.multi_gpu:
        logging.info('run on multi GPU')
        torch.distributed.init_process_group(backend='nccl')

    torch_utils.setup_seed(0)

    output_path = args.output_path
    if args.bert_freeze:
        output_path += '_freeze'
    if args.release:
        output_path += '_release'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = ValidatorLogger(data_path=output_path, log_file='train.log')

    logging.info("loading embedding")
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info("loading dataset")
    train_dataset, test_dataset = get_dataset(args, tokenizer)

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

    best_metric = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info("loading pretrained model")
        model, optimizer, epoch, best_metric = torch_utils.load(args.pretrained_model_path)
        model = model.cpu() if args.use_cpu else model.cuda()
    else:
        logging.info("creating model")
        model = Bert('%s/config.json' % args.pretrained_bert_path,
                     '%s/pytorch_model.bin' % args.pretrained_bert_path,
                     args.bert_freeze)
        model = model.cpu() if args.use_cpu else model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.multi_gpu:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    num_train_steps = int(len(train_dataset) / args.batch_size * args.epoch_size)
    num_warmup_steps = int(num_train_steps * args.lr_warmup_proportion)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_warmup_steps, gamma=args.lr_decay_gamma)

    logging.info("begin training")
    while epoch < args.epoch_size:
        epoch += 1

        train_loss = train(args, train_dataset, train_dataloader, model, optimizer, lr_scheduler)
        test_acc, test_pre, test_rec, test_f1 = evaluate(args, test_dataloader, model)

        logging.info('epoch[%s/%s], train loss: %s' % (epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], test accuracy: %s, precision: %s, recall: %s, f1: %s' % (
            epoch, args.epoch_size, test_acc, test_pre, test_rec, test_f1))

        remark = ''
        if test_f1 > best_metric:
            best_metric = test_f1
            remark = 'best'
            torch_utils.save(output_path, 'best.pth', model, optimizer, epoch, best_metric)

        torch_utils.save(output_path, 'last.pth', model, optimizer, epoch, best_metric)
        model_logger.write(epoch, train_loss, test_acc, test_pre, test_rec, test_f1, remark)

    logging.info("complete training")
    model_logger.draw_plot()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='../data/dataset/')
    parser.add_argument('--pretrained_bert_path', type=str,
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--pretrained_model_path', type=str,
                        default=None)
    parser.add_argument('--output_path', type=str,
                        default='../runtime/validator')
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
                        default=32)
    parser.add_argument('--epoch_size', type=int,
                        default=30)
    parser.add_argument('--learning_rate', type=float,
                        default=5e-5)
    parser.add_argument('--lr_warmup_proportion', type=float,
                        default=0.1)
    parser.add_argument('--lr_decay_gamma', type=float,
                        default=0.9)
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--multi_gpu', type=bool,
                        help='run with: -m torch.distributed.launch',
                        default=False)
    parser.add_argument('--local_rank', type=int,
                        default=0)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
