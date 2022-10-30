import codecs
import json
import logging
from tqdm import tqdm

import faiss
from torch.utils.data import Dataset
from transformers import BertTokenizer

from utils import torch_utils, text_utils, faiss_utils
from utils.data_utils import get_documents_with_paragraphs, TOKEN_CLS, TOKEN_SEP


def create_validate_data(data_path, pretrained_bert_path, retriever_model_path, faiss_path,
                         retrieve_size=10, max_question_len=0, negative_size=10, with_title=True, use_cpu=False):
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
        with codecs.open('%s/%s_validate.txt' % (data_path, split), 'w', 'utf-8') as fout:
            for record in tqdm(records):
                fout.write('%s\n' % (json.dumps(record)))

    logging.info('complete create validate data ')


class ValidateDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_question_len=0, max_context_len=0,
                 with_sep=True, debug=False):
        super(ValidateDataset, self).__init__()
        self.data = []

        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                line = json.loads(line)

                question = text_utils.dbc_to_sbc(str(line['question']))
                # question = tokenizer.tokenize(question)
                question = [ch for ch in question]
                question = [TOKEN_CLS] + question + [TOKEN_SEP]

                question_tokens = tokenizer.convert_tokens_to_ids(question)
                if max_question_len > 0:
                    question_tokens = question_tokens[:max_question_len]
                question_tokens_type_id = [0] * len(question_tokens)

                for para in line['paragraphs']:
                    selected = int(para['selected'])

                    paragraph_tokens = para['chars']
                    if not with_sep:
                        while TOKEN_SEP in paragraph_tokens:
                            paragraph_tokens.remove(TOKEN_SEP)
                    paragraph_tokens = tokenizer.convert_tokens_to_ids(paragraph_tokens)

                    input_tokens = question_tokens + paragraph_tokens
                    input_tokens_type_id = question_tokens_type_id + [1] * len(paragraph_tokens)
                    if max_context_len > 0:
                        input_tokens = input_tokens[:max_context_len]
                        input_tokens_type_id = input_tokens_type_id[:max_context_len]

                    self.data.append([selected, input_tokens, input_tokens_type_id])

                if debug:
                    if len(self.data) >= 100:
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# if __name__ == '__main__':
#     logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
#
#     data_path = '../data/dataset'
#     pretrained_bert_path = '../data/bert/bert-base-chinese/'
#     retriever_model_path = '../runtime/retriever_sep/dpr_ance/last.pth'
#     faiss_path = '../runtime/retriever_sep/dpr_ance/faiss_last'
#     retrieve_size = 30
#     max_question_len = 64
#     negative_size = 10
#
#     create_validate_data(data_path, pretrained_bert_path, retriever_model_path, faiss_path,
#                          retrieve_size, max_question_len, negative_size)
#
#     logging.info('complete')
