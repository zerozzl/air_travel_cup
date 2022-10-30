import os
import json
import codecs
import logging
import xlsxwriter
from tqdm import tqdm
from argparse import ArgumentParser
import faiss
import torch
from transformers import BertTokenizer

from utils.data_utils import get_documents_with_paragraphs
from utils import torch_utils, faiss_utils


def check_retrieve(answers, retrieve_ids, accept_retrieve_size=5):
    retrieve_ids = retrieve_ids[:accept_retrieve_size]
    success = False
    for record in retrieve_ids:
        if success:
            break
        for item in answers:
            if record['content-key'] == item['content-key'] and record['detail'] == item['detail']:
                success = True
                break

    error_list = []
    if not success:
        error_list = retrieve_ids
    return success, error_list


def save_result(output_path, documents, train_error_list, valid_error_list):
    workbook = xlsxwriter.Workbook('%s/error_samples.xlsx' % output_path)

    train_sheet = workbook.add_worksheet('训练')
    train_sheet.activate()
    train_sheet.write(0, 0, '问题')
    train_sheet.write(0, 1, '答案')
    train_sheet.write(0, 2, '预测')
    for idx in range(len(train_error_list)):
        sample = train_error_list[idx]

        for answer in sample[1]:
            if answer['content-key'] in documents:
                document = documents[answer['content-key']]
                for paragraph in document:
                    if paragraph['detail'] == answer['detail']:
                        answer['content'] = ''.join(paragraph['chars'].split())

        for retrieve in sample[2]:
            if retrieve['content-key'] in documents:
                document = documents[retrieve['content-key']]
                for paragraph in document:
                    if paragraph['detail'] == retrieve['detail']:
                        retrieve['content'] = ''.join(paragraph['chars'].split())

        train_sheet.write(idx + 1, 0, sample[0])
        train_sheet.write(idx + 1, 1, str(sample[1]))
        train_sheet.write(idx + 1, 2, str(sample[2]))

    valid_sheet = workbook.add_worksheet('验证')
    valid_sheet.activate()
    valid_sheet.write(0, 0, '问题')
    valid_sheet.write(0, 1, '答案')
    valid_sheet.write(0, 2, '预测')
    for idx in range(len(valid_error_list)):
        sample = valid_error_list[idx]

        for answer in sample[1]:
            if answer['content-key'] in documents:
                document = documents[answer['content-key']]
                for paragraph in document:
                    if paragraph['detail'] == answer['detail']:
                        answer['content'] = ''.join(paragraph['chars'].split())

        for retrieve in sample[2]:
            if retrieve['content-key'] in documents:
                document = documents[retrieve['content-key']]
                for paragraph in document:
                    if paragraph['detail'] == retrieve['detail']:
                        retrieve['content'] = ''.join(paragraph['chars'].split())

        valid_sheet.write(idx + 1, 0, sample[0])
        valid_sheet.write(idx + 1, 1, str(sample[1]))
        valid_sheet.write(idx + 1, 2, str(sample[2]))

    workbook.close()


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.info('loading embedding')
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info('loading documents')
    documents = get_documents_with_paragraphs('%s/content.xlsx' % args.data_path, tokenizer, with_title=True)

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
        error_data = {}
        data_splits = ['train', 'valid']
        for split in data_splits:
            logging.info('Processing %s data' % split)

            error_list = []
            with codecs.open('%s/%s.txt' % (args.data_path, split), 'r', 'utf-8') as fin:
                for line in tqdm(fin):
                    line = line.strip()
                    if line == '':
                        continue

                    line = json.loads(line)
                    question = line['question']
                    answers = line['answer']

                    if len(answers) == 0:
                        continue

                    retrieve_ids, _ = faiss_utils.retrieve_from_faiss(tokenizer, model,
                                                                      faiss_idx_to_id, faiss_index, question,
                                                                      args.retrieve_size, args.max_question_len)
                    retrieve_ids = [json.loads(obj) for obj in retrieve_ids]
                    success, retrieve_list = check_retrieve(answers, retrieve_ids, args.accept_retrieve_size)

                    if not success:
                        error_list.append([question, answers, retrieve_list])

                    if args.debug and len(error_list) >= 5:
                        break

                error_data[split] = error_list

        save_result(args.output_path, documents, error_data['train'], error_data['valid'])

    logging.info('complete')


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
    parser.add_argument('--accept_retrieve_size', type=int,
                        default=5)
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
