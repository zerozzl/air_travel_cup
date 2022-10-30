import os
import codecs
import logging
from argparse import ArgumentParser

import torch
from transformers import BertTokenizer

from ranker.bert import evaluator
from utils import torch_utils


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    torch_utils.setup_seed(0)

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.info("loading embedding")
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info("loading pretrained model")
    model, _, _, _ = torch_utils.load('%s/%s.pth' % (args.model_path, args.model_state))
    model = model.cpu() if args.use_cpu else model.cuda()
    model.eval()

    with torch.no_grad():
        with codecs.open('%s/eval.%s.log' % (output_path, args.model_state), 'w', 'utf-8') as fout:
            data_splits = ['train', 'valid']
            for split in data_splits:
                logging.info('Reading %s data' % split)
                fout.write('Reading %s data\n' % split)

                top_5_recall, top_10_recall, top_50_recall, top_100_recall = evaluator.evaluate(
                    '%s/%s_rank.txt' % (args.data_path, split), tokenizer, model,
                    args.accept_size, args.max_question_len, args.max_context_len,
                    args.batch_size, with_sep=False, use_cpu=args.use_cpu, debug=args.debug)

                logging.info('Top 5 recall %s\nTop 10 recall %s\nTop 50 recall %s\nTop 100 recall %s' % (
                    top_5_recall, top_10_recall, top_50_recall, top_100_recall))
                fout.write('Top 5 recall %s\nTop 10 recall %s\nTop 50 recall %s\nTop 100 recall %s\n' % (
                    top_5_recall, top_10_recall, top_50_recall, top_100_recall))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='../../data/dataset/')
    parser.add_argument('--pretrained_bert_path', type=str,
                        default='../../data/bert/bert-base-chinese/')
    parser.add_argument('--model_path', type=str,
                        default='../../runtime/ranker/bert/pretrain/')
    parser.add_argument('--output_path', type=str,
                        default='../../runtime/ranker/bert/pretrain')
    parser.add_argument('--model_state', type=str,
                        default='last')
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--batch_size', type=int,
                        default=64)
    parser.add_argument('--accept_size', type=int,
                        default=5)
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
