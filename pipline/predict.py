import os
import json
import codecs
import logging
from tqdm import tqdm
from argparse import ArgumentParser
import torch

from pipline import run
from utils import torch_utils


def save_result(output_path, task1_result, task2_result, task3_result):
    with codecs.open('%s/task_1.txt' % output_path, 'w', 'utf-8') as fout:
        for result in task1_result:
            fout.write('%s\n' % json.dumps(result))
    with codecs.open('%s/task_2.txt' % output_path, 'w', 'utf-8') as fout:
        for result in task2_result:
            fout.write('%s\n' % json.dumps(result))
    with codecs.open('%s/task_3.txt' % output_path, 'w', 'utf-8') as fout:
        for result in task3_result:
            fout.write('%s\n' % json.dumps(result))


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    torch_utils.setup_seed(0)

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    tokenizer, documents, documents_with_title, faiss_idx_to_id, faiss_index, \
    retriever_model, ranker_model, reranker_model, validator_model, reader_model = run.init(args)

    task1_result = []
    task2_result = []
    task3_result = []
    with torch.no_grad():
        with codecs.open('%s/test.txt' % args.data_path, 'r', 'utf-8') as fin:
            for line in tqdm(fin):
                line = line.strip()
                if line == '':
                    continue

                question = str(line)

                retrieve_ids, retrieve_scores, validate_ids = run.get_retrieve_and_validate_ids(
                    args, documents_with_title, faiss_idx_to_id, faiss_index, tokenizer,
                    retriever_model, ranker_model, reranker_model, validator_model, question)

                docs_id, paras_id, answers, _, _, _, _ = run.get_pred_result(
                    args, documents, tokenizer, reader_model, question,
                    retrieve_ids, retrieve_scores, validate_ids)

                task1_result.append({'question': question, 'answer': docs_id})
                task2_result.append({'question': question, 'answer': paras_id})
                task3_result.append({'question': question, 'answer': answers})

                if args.debug and len(task1_result) >= 10:
                    break

    save_result(output_path, task1_result, task2_result, task3_result)
    logging.info('complete predict')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='../data/dataset/')
    parser.add_argument('--output_path', type=str,
                        default='../runtime/pipline/faiss')
    parser.add_argument('--pretrained_bert_path', type=str,
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--retriever_model_path', type=str,
                        default='../runtime/retriever/dpr_ance/last.pth')
    parser.add_argument('--faiss_path', type=str,
                        default='../runtime/retriever/dpr_ance/faiss_last')
    parser.add_argument('--ranker_model_path', type=str,
                        default='../runtime/ranker/bert/pretrain/best.pth')
    parser.add_argument('--reranker_model_path', type=str,
                        default='../runtime/reranker/last.pth')
    parser.add_argument('--validator_model_path', type=str,
                        default='../runtime/validator_old/best.pth')
    parser.add_argument('--reader_model_path', type=str,
                        default='../runtime/reader/best.pth')
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--retrieve_size', type=int,
                        default=100)
    parser.add_argument('--batch_size', type=int,
                        default=64)
    parser.add_argument('--use_ranker', type=bool,
                        default=False)
    parser.add_argument('--use_reranker', type=bool,
                        default=False)
    parser.add_argument('--accept_retrieve_size', type=int,
                        default=5)
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
