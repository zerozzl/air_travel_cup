import os
import json
import codecs
import logging
from tqdm import tqdm
from argparse import ArgumentParser
import faiss
import torch
from transformers import BertTokenizer

from retriever import evaluator
from utils import torch_utils, faiss_utils


def filter_by_threshold(retrieve_ids, retrieve_scos, score_threshold):
    filter_result = []
    for idx in range(len(retrieve_ids)):
        score = retrieve_scos[idx]
        if score >= score_threshold:
            filter_result.append(retrieve_ids[idx])
    return filter_result


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    logging.getLogger('elasticsearch').setLevel(logging.ERROR)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.info('loading embedding')
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info('loading pretrained model')
    model_path = '%s/%s.pth' % (args.model_path, args.model_state)
    model, _, _, _ = torch_utils.load(model_path)
    model = model.cpu() if args.use_cpu else model.cuda()
    model.eval()

    logging.info('loading faiss index')
    faiss_id_path = '%s/faiss_%s.id' % (args.model_path, args.model_state)
    faiss_data_path = '%s/faiss_%s.data' % (args.model_path, args.model_state)
    faiss_idx_to_id = faiss_utils.load_faiss_id(faiss_id_path)
    faiss_index = faiss.read_index(faiss_data_path)
    gpu_res = faiss.StandardGpuResources()
    faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss_index)

    with torch.no_grad():
        with codecs.open('%s/faiss_%s.log' % (output_path, args.model_state), 'w', 'utf-8') as fout:
            data_splits = ['train', 'valid']
            for split in data_splits:
                total_num = 0
                top1_num = 0
                top3_num = 0
                top5_num = 0
                top10_num = 0
                top50_num = 0
                top100_num = 0
                top500_num = 0
                top1000_num = 0
                data_split_path = '%s/%s.txt' % (args.data_path, split)

                logging.info('Reading: %s' % data_split_path)
                with codecs.open(data_split_path, 'r', 'utf-8') as fin:
                    for line in tqdm(fin):
                        line = line.strip()
                        if line == '':
                            continue

                        line = json.loads(line)
                        question = line['question']
                        answers = line['answer']

                        retrieve_ids, retrieve_scos = faiss_utils.retrieve_from_faiss(tokenizer, model,
                                                                                      faiss_idx_to_id, faiss_index,
                                                                                      question,
                                                                                      args.retrieve_size,
                                                                                      args.max_question_len,
                                                                                      args.use_cpu)
                        retrieve_ids = [json.loads(obj) for obj in retrieve_ids]
                        retrieve_ids = filter_by_threshold(retrieve_ids, retrieve_scos, args.score_threshold)

                        total_num += 1
                        top1_num += evaluator.evaluate(answers, retrieve_ids[:1])
                        top3_num += evaluator.evaluate(answers, retrieve_ids[:3])
                        top5_num += evaluator.evaluate(answers, retrieve_ids[:5])
                        top10_num += evaluator.evaluate(answers, retrieve_ids[:10])
                        top50_num += evaluator.evaluate(answers, retrieve_ids[:50])
                        top100_num += evaluator.evaluate(answers, retrieve_ids[:100])
                        top500_num += evaluator.evaluate(answers, retrieve_ids[:500])
                        top1000_num += evaluator.evaluate(answers, retrieve_ids[:1000])

                log_text = '%s data\n' % split
                log_text = log_text + 'top 1 recall %.3f\n' % (top1_num / total_num)
                log_text = log_text + 'top 3 recall %.3f\n' % (top3_num / total_num)
                log_text = log_text + 'top 5 recall %.3f\n' % (top5_num / total_num)
                log_text = log_text + 'top 10 recall %.3f\n' % (top10_num / total_num)
                log_text = log_text + 'top 50 recall %.3f\n' % (top50_num / total_num)
                log_text = log_text + 'top 100 recall %.3f\n' % (top100_num / total_num)
                log_text = log_text + 'top 500 recall %.3f\n' % (top500_num / total_num)
                log_text = log_text + 'top 1000 recall %.3f\n' % (top1000_num / total_num)

                logging.info(log_text)
                fout.write('%s\n' % log_text)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='../data/dataset/')
    parser.add_argument('--pretrained_bert_path', type=str,
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--model_path', type=str,
                        default='../runtime/retriever/dpr_ance/')
    parser.add_argument('--model_state', type=str,
                        default='last')
    parser.add_argument('--output_path', type=str,
                        default='../runtime/retriever/dpr_ance/')
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--retrieve_size', type=int,
                        default=1000)
    parser.add_argument('--score_threshold', type=int,
                        default=0)
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
