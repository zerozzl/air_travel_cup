import json
import codecs
import faiss
import logging
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from ranker.bert import evaluator
from utils import torch_utils, text_utils, faiss_utils
from utils.data_utils import get_documents_with_paragraphs, TOKEN_CLS, TOKEN_SEP, TOKEN_MASK, COLBERT_Q_SPE, \
    COLBERT_D_SPE


def create_data_from_retriever(data_path, pretrained_bert_path, retriever_model_path, faiss_path,
                               retrieve_size=1000, max_question_len=0, negative_size=100, with_title=True,
                               use_cpu=False):
    logging.info('loading embedding')
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % pretrained_bert_path)

    logging.info('reading data from %s' % data_path)
    documents = get_documents_with_paragraphs('%s/content.xlsx' % data_path, tokenizer, with_title=with_title)

    logging.info('loading retriever model: %s' % retriever_model_path)
    retriever_model, _, _, _ = torch_utils.load(retriever_model_path)
    retriever_model = retriever_model.cpu() if use_cpu else retriever_model.cuda()
    retriever_model.eval()

    logging.info('loading faiss index: %s' % faiss_path)
    faiss_idx_to_id = faiss_utils.load_faiss_id('%s.id' % faiss_path)
    faiss_index = faiss.read_index('%s.data' % faiss_path)
    gpu_res = faiss.StandardGpuResources()
    faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss_index)

    data_splits = ['train', 'valid']
    for split in data_splits:
        logging.info('processing %s data' % split)

        records = []
        with codecs.open('%s/%s.txt' % (data_path, split), 'r', 'utf-8') as fin:
            for line in tqdm(fin):
                line = line.strip()
                if line == '':
                    continue

                line = json.loads(line)
                question = line['question']
                paragraphs = []
                for answer in line['answer']:
                    document = documents[answer['content-key']]
                    for para in document:
                        if para['detail'] == answer['detail']:
                            paragraphs.append({
                                'content-key': para['content-key'],
                                'detail': para['detail'],
                                'chars': para['chars'].split(),
                                'tokens': para['tokens'],
                                'selected': True,
                            })

                retrieve_ids, _ = faiss_utils.retrieve_from_faiss(tokenizer, retriever_model,
                                                                  faiss_idx_to_id, faiss_index, question,
                                                                  retrieve_size=retrieve_size,
                                                                  max_question_len=max_question_len,
                                                                  use_cpu=use_cpu)

                add_negative_size = 0
                for retrieve_id in retrieve_ids:
                    if add_negative_size >= negative_size:
                        break

                    retrieve_id = json.loads(retrieve_id)
                    selected = False
                    for answer in paragraphs:
                        if (retrieve_id['content-key'] == answer['content-key']) and (
                                retrieve_id['detail'] == answer['detail']):
                            selected = True
                            break
                    if not selected:
                        document = documents[retrieve_id['content-key']]
                        for para in document:
                            if para['detail'] == retrieve_id['detail']:
                                paragraphs.append({
                                    'content-key': para['content-key'],
                                    'detail': para['detail'],
                                    'chars': para['chars'].split(),
                                    'tokens': para['tokens'],
                                    'selected': False,
                                })

                                add_negative_size += 1

                records.append({
                    'question': line['question'],
                    'aim': line['aim'],
                    'paragraphs': paragraphs
                })

        logging.info('saving %s data' % split)
        with codecs.open('%s/%s_rank.txt' % (data_path, split), 'w', 'utf-8') as fout:
            for record in tqdm(records):
                fout.write('%s\n' % (json.dumps(record)))

    logging.info('complete create rank data ')


def create_data_from_ranker(data_path, output_path, tokenizer, model, negative_size=4,
                            max_question_len=0, max_context_len=0, batch_size=32, use_cpu=False, debug=False):
    with torch.no_grad():
        data_splits = ['train', 'valid']
        for split in data_splits:
            logging.info('processing %s data' % split)

            records = []
            with codecs.open('%s/%s_rank.txt' % (data_path, split), 'r', 'utf-8') as fin:
                for line in tqdm(fin):
                    if debug and len(records) >= 10:
                        break
                    line = line.strip()
                    if line == '':
                        continue

                    line = json.loads(line)
                    question = line['question']
                    paragraphs = line['paragraphs']

                    para_sorted, _ = evaluator.check_question(tokenizer, model, question, paragraphs,
                                                              1000, max_question_len, max_context_len, batch_size,
                                                              use_cpu=use_cpu)

                    labels = np.array([int(para['selected']) for para in paragraphs])
                    labels = list(np.where(labels == 1)[0])

                    paragraphs_update = []
                    for idx in labels:
                        paragraphs_update.append(paragraphs[idx])

                    add_negative_size = 0
                    for idx in para_sorted:
                        if add_negative_size >= negative_size:
                            break
                        if idx in labels:
                            continue

                        paragraphs_update.append(paragraphs[idx])
                        add_negative_size += 1

                    records.append({
                        'question': line['question'],
                        'aim': line['aim'],
                        'paragraphs': paragraphs_update
                    })

            logging.info('saving %s data' % split)
            with codecs.open('%s/%s_rank_finetune.txt' % (output_path, split), 'w', 'utf-8') as fout:
                for record in tqdm(records):
                    fout.write('%s\n' % (json.dumps(record)))

    logging.info('complete create rank data ')


def read_data(data_path, debug=False):
    data = []
    with codecs.open(data_path, 'r', 'utf-8') as fin:
        for line in fin:
            line = json.loads(line)

            question = str(line['question'])

            positive = []
            negative = []
            for para in line['paragraphs']:
                selected = para['selected']
                # tokens = para['tokens']
                tokens = para['chars']
                if selected:
                    positive.append(tokens)
                else:
                    negative.append(tokens)

            data.append({
                'question': question,
                'positive': positive,
                'negative': negative
            })

            if debug:
                if len(data) >= 100:
                    break
    return data


class RankBertDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_question_len=0, max_context_len=0,
                 epoch=0, negative_batch_size=0, with_sep=False):
        super(RankBertDataset, self).__init__()
        self.data = []

        for line in dataset:
            question = text_utils.dbc_to_sbc(str(line['question']))
            # question = tokenizer.tokenize(question)
            question = [ch for ch in question]
            question = [TOKEN_CLS] + question + [TOKEN_SEP]

            question_tokens = tokenizer.convert_tokens_to_ids(question)
            if max_question_len > 0:
                question_tokens = question_tokens[:max_question_len]
            question_tokens_type_id = [0] * len(question_tokens)

            positive = []
            for paragraph in line['positive']:
                if not with_sep:
                    while TOKEN_SEP in paragraph:
                        paragraph.remove(TOKEN_SEP)

                paragraph = tokenizer.convert_tokens_to_ids(paragraph)
                paragraph_tokens_type_id = [1] * len(paragraph)

                if max_context_len > 0:
                    paragraph = paragraph[:max_context_len - len(question)]
                    paragraph_tokens_type_id = paragraph_tokens_type_id[:max_context_len - len(question)]

                positive.append([question_tokens + paragraph, question_tokens_type_id + paragraph_tokens_type_id])

            negative = []
            negative_list = line['negative']
            if negative_batch_size > 0:
                negative_list = self.get_sub_list(epoch, negative_batch_size, negative_list)
            for paragraph in negative_list:
                if not with_sep:
                    while TOKEN_SEP in paragraph:
                        paragraph.remove(TOKEN_SEP)

                paragraph = tokenizer.convert_tokens_to_ids(paragraph)
                paragraph_tokens_type_id = [1] * len(paragraph)

                if max_context_len > 0:
                    paragraph = paragraph[:max_context_len - len(question)]
                    paragraph_tokens_type_id = paragraph_tokens_type_id[:max_context_len - len(question)]

                negative.append([question_tokens + paragraph, question_tokens_type_id + paragraph_tokens_type_id])

            for pos_idx in range(len(positive)):
                self.data.append([positive[pos_idx], negative])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_sub_list(self, epoch, batch_size, data_list):
        begin = (epoch - 1) * batch_size
        end = epoch * batch_size

        if begin >= len(data_list):
            begin = begin % len(data_list)
            end = end % len(data_list)

            if end == 0:
                end = len(data_list)
                sublist = data_list[begin:end]
            elif end <= begin:
                sublist = data_list[begin:len(data_list)] + data_list[0:end]
            else:
                sublist = data_list[begin:end]
        elif end >= len(data_list):
            sublist = data_list[begin:len(data_list)] + data_list[0:end - len(data_list)]
        else:
            sublist = data_list[begin:end]

        return sublist


class RankColBertDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_question_len=0, max_context_len=0,
                 epoch=0, negative_batch_size=0, with_sep=False):
        super(RankColBertDataset, self).__init__()
        self.data = []

        for line in dataset:
            question = text_utils.dbc_to_sbc(str(line['question']))
            # question = tokenizer.tokenize(question)
            question = [ch for ch in question]
            question = [TOKEN_CLS, COLBERT_Q_SPE] + question
            if max_question_len > 0:
                question = question[:max_question_len] + [TOKEN_MASK] * (
                        max_question_len - len(question))
            question_tokens = tokenizer.convert_tokens_to_ids(question)
            question_tokens_type_id = [0] * len(question_tokens)

            positive = []
            for paragraph in line['positive']:
                if not with_sep:
                    while TOKEN_SEP in paragraph:
                        paragraph.remove(TOKEN_SEP)

                paragraph = [TOKEN_CLS, COLBERT_D_SPE] + paragraph
                if max_context_len > 0:
                    paragraph = paragraph[:max_context_len]
                paragraph = tokenizer.convert_tokens_to_ids(paragraph)
                paragraph_tokens_type_id = [1] * len(paragraph)

                positive.append([paragraph, paragraph_tokens_type_id])

            negative = []
            negative_list = line['negative']
            if negative_batch_size > 0:
                negative_list = self.get_sub_list(epoch, negative_batch_size, negative_list)
            for paragraph in negative_list:
                if not with_sep:
                    while TOKEN_SEP in paragraph:
                        paragraph.remove(TOKEN_SEP)

                paragraph = [TOKEN_CLS, COLBERT_D_SPE] + paragraph
                if max_context_len > 0:
                    paragraph = paragraph[:max_context_len]
                paragraph = tokenizer.convert_tokens_to_ids(paragraph)
                paragraph_tokens_type_id = [1] * len(paragraph)

                negative.append([paragraph, paragraph_tokens_type_id])

            for pos_idx in range(len(positive)):
                self.data.append([[question_tokens, question_tokens_type_id], positive[pos_idx], negative])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_sub_list(self, epoch, batch_size, data_list):
        begin = (epoch - 1) * batch_size
        end = epoch * batch_size

        if begin >= len(data_list):
            begin = begin % len(data_list)
            end = end % len(data_list)

            if end == 0:
                end = len(data_list)
                sublist = data_list[begin:end]
            elif end <= begin:
                sublist = data_list[begin:len(data_list)] + data_list[0:end]
            else:
                sublist = data_list[begin:end]
        elif end >= len(data_list):
            sublist = data_list[begin:len(data_list)] + data_list[0:end - len(data_list)]
        else:
            sublist = data_list[begin:end]

        return sublist

# if __name__ == '__main__':
#     logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
#
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
#     data_path = '../data/dataset'
#     pretrained_bert_path = '../data/bert/bert-base-chinese/'
#     retriever_model_path = '../runtime/retriever/dpr_ance/last.pth'
#     ranker_model_path = '../runtime/ranker/pretrain/last.pth'
#     faiss_path = '../runtime/retriever/dpr_ance/faiss_last'
#     retrieve_size = 1000
#     max_question_len = 64
#     max_context_len = 512
#     retriever_negative_size = 100
#     ranker_negative_size = 4
#     batch_size = 64
#     with_title = True
#     use_cpu = False
#
#     create_data_from_retriever(data_path, pretrained_bert_path, retriever_model_path, faiss_path,
#                                retrieve_size, max_question_len, retriever_negative_size,
#                                with_title=with_title, use_cpu=use_cpu)
#
#     # logging.info("loading embedding")
#     # tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % pretrained_bert_path)
#     #
#     # logging.info("loading pretrained model")
#     # model, _, _, _ = torch_utils.load(ranker_model_path)
#     # model = model.cpu() if use_cpu else model.cuda()
#     # model.eval()
#     #
#     # create_data_from_ranker(data_path, data_path, tokenizer, model, ranker_negative_size,
#     #                         max_question_len=max_question_len, max_context_len=max_context_len, batch_size=batch_size,
#     #                         use_cpu=use_cpu, debug=False)
