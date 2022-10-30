import os
import json
import codecs
import logging
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from torch.nn.utils import rnn
import torch.nn.functional as F
from transformers import BertTokenizer

from validator import evaluator
from utils.data_utils import TOKEN_CLS, TOKEN_SEP
from utils import torch_utils, text_utils


def get_input(args, tokenizer, question, paragraphs):
    tokens = []
    tokens_type_id = []
    labels = []

    question = text_utils.dbc_to_sbc(str(question))
    question = tokenizer.tokenize(question)
    question = [TOKEN_CLS] + question + [TOKEN_SEP]

    question_tokens = tokenizer.convert_tokens_to_ids(question)
    if args.max_question_len > 0:
        question_tokens = question_tokens[:args.max_question_len]
    question_tokens_type_id = [0] * len(question_tokens)

    for para in paragraphs:
        selected = int(para['selected'])

        paragraph_tokens = para['tokens']
        paragraph_tokens = tokenizer.convert_tokens_to_ids(paragraph_tokens)

        input_tokens = question_tokens + paragraph_tokens
        input_tokens_type_id = question_tokens_type_id + [1] * len(paragraph_tokens)
        if args.max_context_len > 0:
            input_tokens = input_tokens[:args.max_context_len]
            input_tokens_type_id = input_tokens_type_id[:args.max_context_len]

        tokens.append(torch.LongTensor(input_tokens))
        tokens_type_id.append(torch.LongTensor(input_tokens_type_id))
        labels.append(selected)

    tokens = rnn.pad_sequence(tokens, batch_first=True)
    tokens_type_id = rnn.pad_sequence(tokens_type_id, batch_first=True)
    tokens_mask = (tokens > 0).int()

    tokens = tokens.cpu() if args.use_cpu else tokens.cuda()
    tokens_type_id = tokens_type_id.cpu() if args.use_cpu else tokens_type_id.cuda()
    tokens_mask = tokens_mask.cpu() if args.use_cpu else tokens_mask.cuda()

    return labels, tokens, tokens_type_id, tokens_mask


def get_label(score_list, threshold=0):
    result = []
    for idx in range(len(score_list)):
        if score_list[idx][1] >= threshold:
            result.append(1)
        else:
            result.append(0)
    return result


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    torch_utils.setup_seed(0)

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.info('loading embedding')
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info('loading model')
    model, _, _, _ = torch_utils.load(args.model_path)
    model = model.cpu() if args.use_cpu else model.cuda()
    model.eval()

    with codecs.open('%s/eval.log' % args.output_path, 'w', 'utf-8') as fout:
        with torch.no_grad():
            data_splits = ['train', 'valid']
            for split in data_splits:
                pred_answers = []
                gold_answers = []
                with codecs.open('%s/%s_validate.txt' % (args.data_path, split), 'r', 'utf-8') as fin:
                    for line in tqdm(fin):
                        line = line.strip()
                        if line == '':
                            continue

                        line = json.loads(line)
                        question = line['question']
                        paragraphs = line['paragraphs']

                        labels, tokens, tokens_type_id, tokens_mask = get_input(args, tokenizer, question, paragraphs)
                        logits = model(tokens, tokens_type_id, tokens_mask)
                        logits = F.softmax(logits, dim=1)
                        preds = torch.argmax(logits, dim=1)
                        preds = preds.cpu().numpy()

                        gold_answers.extend(labels)
                        pred_answers.extend(preds)

                        if args.debug and len(pred_answers) >= 100:
                            break

                acc, pre, rec, f1 = evaluator.evaluate(gold_answers, pred_answers)
                logging.info('%s accuracy: %s, precision: %s, recall: %s, f1: %s' % (
                    split, acc, pre, rec, f1))
                fout.write('%s accuracy: %s, precision: %s, recall: %s, f1: %s\n' % (
                    split, acc, pre, rec, f1))

    logging.info('complete testing')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='../data/dataset/')
    parser.add_argument('--pretrained_bert_path', type=str,
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--model_path', type=str,
                        default='../runtime/validator/best.pth')
    parser.add_argument('--output_path', type=str,
                        default='../runtime/validator/')
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
