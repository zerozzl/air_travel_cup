import os
import logging
import numpy as np
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import rnn
import torch.nn.functional as F
from transformers import BertTokenizer

from retriever.dataloader import DprDataset, create_dpr_from_faiss
from models.Retriever import DprBert
from utils import torch_utils, faiss_utils
from utils.log_utils import RetrieverLogger


def get_dataset(args, data_path, tokenizer):
    train_dataset = DprDataset('%s/train_dpr.txt' % data_path, tokenizer,
                               args.max_question_len, args.max_context_len, with_sep=False, debug=args.debug)
    test_dataset = DprDataset('%s/valid_dpr.txt' % data_path, tokenizer,
                              args.max_question_len, args.max_context_len, with_sep=False, debug=args.debug)

    if args.release:
        train_dataset.data.extend(test_dataset.data)

    return train_dataset, test_dataset


def get_dataloader(args, train_dataset, test_dataset):
    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      sampler=DistributedSampler(train_dataset, shuffle=True))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     sampler=DistributedSampler(test_dataset, shuffle=False))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     shuffle=False)
    return train_dataloader, test_dataloader


def data_collate_fn(data):
    exists_ids = set()

    question_tokens = []
    question_tokens_type_id = []
    paragraph_tokens = []
    paragraph_tokens_type_id = []
    for record in data:
        if record[0] in exists_ids:
            continue
        exists_ids.add(record[0])

        question_tokens.append(torch.LongTensor(np.array(record[1])))
        question_tokens_type_id.append(torch.LongTensor(np.array(record[2])))

        for paragraph in record[3]:
            paragraph_tokens.append(torch.LongTensor(np.array(paragraph[0])))
            paragraph_tokens_type_id.append(torch.LongTensor(np.array(paragraph[1])))

    labels = [idx * 2 for idx in range(len(question_tokens))]
    labels = torch.LongTensor(labels)

    question_tokens = rnn.pad_sequence(question_tokens, batch_first=True)
    question_tokens_type_id = rnn.pad_sequence(question_tokens_type_id, batch_first=True)
    paragraph_tokens = rnn.pad_sequence(paragraph_tokens, batch_first=True)
    paragraph_tokens_type_id = rnn.pad_sequence(paragraph_tokens_type_id, batch_first=True)

    return labels, question_tokens, question_tokens_type_id, paragraph_tokens, paragraph_tokens_type_id


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()

        labels, question_tokens, question_tokens_type_id, paragraph_tokens, paragraph_tokens_type_id = data

        labels = labels.cpu() if args.use_cpu else labels.cuda()
        question_tokens = question_tokens.cpu() if args.use_cpu else question_tokens.cuda()
        question_tokens_type_id = question_tokens_type_id.cpu() if args.use_cpu else question_tokens_type_id.cuda()
        paragraph_tokens = paragraph_tokens.cpu() if args.use_cpu else paragraph_tokens.cuda()
        paragraph_tokens_type_id = paragraph_tokens_type_id.cpu() if args.use_cpu else paragraph_tokens_type_id.cuda()

        scores = model(question_tokens, question_tokens_type_id, paragraph_tokens, paragraph_tokens_type_id)
        loss = F.cross_entropy(scores, labels)
        loss = loss.mean()

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    loss_sum = loss_sum / len(dataset)
    return loss_sum


def evaluate(args, dataloader, model):
    model.eval()

    if args.multi_gpu:
        model = model.module

    total_num = 0
    correct_num = 0
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            labels, question_tokens, question_tokens_type_id, paragraph_tokens, paragraph_tokens_type_id = data

            question_tokens = question_tokens.cpu() if args.use_cpu else question_tokens.cuda()
            question_tokens_type_id = question_tokens_type_id.cpu() if args.use_cpu else question_tokens_type_id.cuda()
            paragraph_tokens = paragraph_tokens.cpu() if args.use_cpu else paragraph_tokens.cuda()
            paragraph_tokens_type_id = paragraph_tokens_type_id.cpu() if args.use_cpu else paragraph_tokens_type_id.cuda()

            scores = model(question_tokens, question_tokens_type_id, paragraph_tokens, paragraph_tokens_type_id)
            scores = F.softmax(scores, dim=1)
            preds = torch.argmax(scores, dim=1)

            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()

            total_num += len(preds)
            correct_num += (preds == labels).sum()

    accuracy = float(correct_num) / total_num
    return accuracy


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch_utils.setup_seed(0)

    if args.multi_gpu:
        logging.info('run on multi GPU')
        torch.distributed.init_process_group(backend='nccl')

    output_path = args.output_path
    if args.bert_freeze:
        output_path += '_freeze'
    if args.ance:
        output_path += '_ance'
    if args.release:
        output_path += '_release'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = RetrieverLogger(data_path=output_path, log_file='train.log')

    logging.info('loading embedding')
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info('loading dataset')
    train_dataset, test_dataset = get_dataset(args, args.data_path, tokenizer)
    train_dataloader, test_dataloader = get_dataloader(args, train_dataset, test_dataset)

    best_metric = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info('loading pretrained model')
        model, optimizer, epoch, best_metric = torch_utils.load(args.pretrained_model_path)
        model = model.cpu() if args.use_cpu else model.cuda()
    else:
        logging.info('creating model')
        model = DprBert('%s/config.json' % args.pretrained_bert_path,
                        '%s/pytorch_model.bin' % args.pretrained_bert_path,
                        args.bert_freeze)
        model = model.cpu() if args.use_cpu else model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.multi_gpu:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    logging.info('begin training')
    while epoch < args.epoch_size:
        epoch += 1

        if args.ance and (epoch % args.rebuild_epoch_size == 0):
            logging.info('reloading dataset')
            faiss_utils.build_faiss_data(args.data_path, output_path, 'last', tokenizer, model,
                                         max_context_len=args.max_context_len, batch_size=4096,
                                         with_title=True, use_cpu=args.use_cpu, debug=args.debug)
            create_dpr_from_faiss(args.data_path, output_path, output_path, 'last', tokenizer, model,
                                  args.max_question_len, args.rebuild_negative_size,
                                  with_title=True, use_cpu=args.use_cpu)

            train_dataset, test_dataset = get_dataset(args, output_path, tokenizer)
            train_dataloader, test_dataloader = get_dataloader(args, train_dataset, test_dataset)

        num_train_steps = int(len(train_dataset) / args.batch_size * args.epoch_size)
        num_warmup_steps = int(num_train_steps * args.lr_warmup_proportion)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_warmup_steps, gamma=args.lr_decay_gamma)

        train_loss = train(args, train_dataset, train_dataloader, model, optimizer, lr_scheduler)
        test_acc = evaluate(args, test_dataloader, model)

        logging.info('epoch[%s/%s], train loss: %s' % (epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], test accuracy: %s' % (epoch, args.epoch_size, test_acc))

        remark = ''
        if test_acc > best_metric:
            best_metric = test_acc
            remark = 'best'
            torch_utils.save(output_path, 'best.pth', model, optimizer, epoch, best_metric)

        torch_utils.save(output_path, 'last.pth', model, optimizer, epoch, best_metric)
        model_logger.write(epoch, train_loss, test_acc, remark)

    model_logger.draw_plot()
    logging.info('complete training')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='../data/dataset/')
    parser.add_argument('--pretrained_bert_path', type=str,
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--pretrained_model_path', type=str,
                        default=None)
    parser.add_argument('--output_path', type=str,
                        default='../runtime/retriever/dpr')
    parser.add_argument('--bert_freeze', type=bool,
                        default=False)
    parser.add_argument('--ance', type=bool,
                        default=True)
    parser.add_argument('--release', type=bool,
                        default=False)
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--batch_size', type=int,
                        default=16)
    parser.add_argument('--epoch_size', type=int,
                        default=100)
    parser.add_argument('--rebuild_epoch_size', type=int,
                        default=5)
    parser.add_argument('--rebuild_negative_size', type=int,
                        default=5)
    parser.add_argument('--learning_rate', type=float,
                        default=1e-5)
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
