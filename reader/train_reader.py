import os
import logging
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import rnn
import torch.nn.functional as F
from transformers import BertTokenizer

from models.Reader import Bert
from reader.dataloader import MrDataset
from reader import evaluator
from utils import torch_utils
from utils.log_utils import ReaderLogger
from utils.data_utils import read_document


def get_dataset(args, documents, tokenizer):
    train_dataset = MrDataset('%s/train.txt' % args.data_path, documents, tokenizer,
                              max_context_len=args.max_context_len, max_question_len=args.max_question_len,
                              do_to_id=True, debug=args.debug)
    test_dataset = MrDataset('%s/valid.txt' % args.data_path, documents, tokenizer,
                             max_context_len=args.max_context_len, max_question_len=args.max_question_len,
                             do_to_id=True, debug=args.debug)

    if args.release:
        train_dataset.data.extend(test_dataset.data)

    return train_dataset, test_dataset


def data_collate_fn(data):
    data = np.array(data)

    labels = torch.LongTensor(np.array(data[:, 0].tolist()))

    input_tokens = data[:, 1].tolist()
    input_tokens = [torch.LongTensor(np.array(item)) for item in input_tokens]
    input_tokens = rnn.pad_sequence(input_tokens, batch_first=True)

    input_tokens_type_id = data[:, 2].tolist()
    input_tokens_type_id = [torch.LongTensor(np.array(item)) for item in input_tokens_type_id]
    input_tokens_type_id = rnn.pad_sequence(input_tokens_type_id, batch_first=True)

    texts = data[:, 3].tolist()
    answers = data[:, 4].tolist()

    input_tokens_mask = (input_tokens > 0).int()

    return labels, input_tokens, input_tokens_type_id, input_tokens_mask, texts, answers


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        labels, input_tokens, input_tokens_type_id, input_tokens_mask, _, _ = data

        labels = labels.cpu() if args.use_cpu else labels.cuda()
        input_tokens = input_tokens.cpu() if args.use_cpu else input_tokens.cuda()
        input_tokens_type_id = input_tokens_type_id.cpu() if args.use_cpu else input_tokens_type_id.cuda()
        input_tokens_mask = input_tokens_mask.cpu() if args.use_cpu else input_tokens_mask.cuda()

        starts, ends = model(input_tokens, input_tokens_type_id, input_tokens_mask)
        labels_start = labels[:, 0]
        labels_end = labels[:, 1]

        loss_start = F.cross_entropy(starts, labels_start)
        loss_end = F.cross_entropy(ends, labels_end)

        loss = loss_start + loss_end
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
            _, input_tokens, input_tokens_type_id, input_tokens_mask, texts, answers = data

            input_tokens = input_tokens.cpu() if args.use_cpu else input_tokens.cuda()
            input_tokens_type_id = input_tokens_type_id.cpu() if args.use_cpu else input_tokens_type_id.cuda()
            input_tokens_mask = input_tokens_mask.cpu() if args.use_cpu else input_tokens_mask.cuda()

            starts, ends = model(input_tokens, input_tokens_type_id, input_tokens_mask)
            starts = torch.argmax(starts, dim=1)
            ends = torch.argmax(ends, dim=1)
            starts = starts.cpu().numpy()
            ends = ends.cpu().numpy()

            for idx in range(len(starts)):
                ctx = texts[idx]
                start = starts[idx]
                end = ends[idx]

                if end >= start:
                    pred_answers.append(''.join([str(ch) for ch in ctx[start:end + 1]]))
                else:
                    pred_answers.append('')

                gold_answers.append(answers[idx])

    f1_score, em_score, _, _ = evaluator.evaluate(gold_answers, pred_answers)
    return f1_score, em_score


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

    model_logger = ReaderLogger(data_path=output_path, log_file='train.log')

    logging.info('loading embedding')
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info('loading dataset')
    documents = read_document('%s/content.xlsx' % args.data_path)
    documents = {record['content-key']: record for record in documents}
    train_dataset, test_dataset = get_dataset(args, documents, tokenizer)

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

    best_metric = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info('loading pretrained model')
        model, optimizer, epoch, best_metric = torch_utils.load(args.pretrained_model_path)
        model = model.cpu() if args.use_cpu else model.cuda()
    else:
        logging.info('creating model')
        model = Bert('%s/config.json' % args.pretrained_bert_path, '%s/pytorch_model.bin' % args.pretrained_bert_path,
                     args.bert_freeze)
        model = model.cpu() if args.use_cpu else model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.multi_gpu:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    num_train_steps = int(len(train_dataset) / args.batch_size * args.epoch_size)
    num_warmup_steps = int(num_train_steps * args.lr_warmup_proportion)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_warmup_steps, gamma=args.lr_decay_gamma)

    logging.info('begin training')
    while epoch < args.epoch_size:
        epoch += 1

        train_loss = train(args, train_dataset, train_dataloader, model, optimizer, lr_scheduler)
        test_f1, test_em = evaluate(args, test_dataloader, model)

        logging.info('epoch[%s/%s], train loss: %s' % (epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], test f1: %s, em: %s' % (epoch, args.epoch_size, test_f1, test_em))

        remark = ''
        if test_f1 > best_metric:
            best_metric = test_f1
            remark = 'best'
            torch_utils.save(output_path, 'best.pth', model, optimizer, epoch, best_metric)

        torch_utils.save(output_path, 'last.pth', model, optimizer, epoch, best_metric)
        model_logger.write(epoch, train_loss, test_f1, test_em, remark)

    logging.info('complete training')
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
                        default='../runtime/reader')
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
