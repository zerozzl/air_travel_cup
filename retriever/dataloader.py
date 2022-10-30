import codecs
import json
import logging
from tqdm import tqdm
from elasticsearch import Elasticsearch
import faiss
import torch

from utils import text_utils, elasticsearch_utils, faiss_utils
from utils.data_utils import read_document, get_paragraphs, get_documents_with_paragraphs, \
    TOKEN_CLS, TOKEN_SEP


def create_document_index(es_host, es_index, es_doc_type):
    es = Elasticsearch([{'host': es_host, 'port': 9200}])

    mapping = {
        "mappings": {
            es_doc_type: {
                "properties": {
                    "key": {
                        "type": "keyword"
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "ik_smart",
                        "fields": {
                            "cn": {
                                "type": "text",
                                "analyzer": "ik_smart"
                            },
                            "en": {
                                "type": "text",
                                "analyzer": "english"
                            }
                        }
                    },
                    "chars": {
                        "type": "text",
                        "analyzer": "ik_smart",
                        "fields": {
                            "cn": {
                                "type": "text",
                                "analyzer": "ik_smart"
                            },
                            "en": {
                                "type": "text",
                                "analyzer": "english"
                            }
                        }
                    },
                    "bigrams": {
                        "type": "text",
                        "analyzer": "ik_smart",
                        "fields": {
                            "cn": {
                                "type": "text",
                                "analyzer": "ik_smart"
                            },
                            "en": {
                                "type": "text",
                                "analyzer": "english"
                            }
                        }
                    },
                    "words": {
                        "type": "text",
                        "analyzer": "ik_smart",
                        "fields": {
                            "cn": {
                                "type": "text",
                                "analyzer": "ik_smart"
                            },
                            "en": {
                                "type": "text",
                                "analyzer": "english"
                            }
                        }
                    }
                }
            }
        }
    }

    if es.indices.exists(index=es_index):
        logging.info('index %s already exists' % es_index)
    else:
        result = es.indices.create(index=es_index, body=mapping)
        logging.info('create index %s' % es_index)
        logging.info(result)


def insert_document_index(es_host, es_index, es_doc_type, data_path, tokenizer, with_title=True):
    logging.info('reading %s' % data_path)
    documents = read_document(data_path)

    es = Elasticsearch([{'host': es_host, 'port': 9200}])
    logging.info('create es index %s' % es_index)
    create_document_index(es_host, es_index, es_doc_type)

    for document in tqdm(documents):
        chars = []
        tokens = []

        title = str(document['title'])
        if with_title:
            chars.extend([ch for ch in title] + [TOKEN_SEP])
            tokens.extend(tokenizer.tokenize(title) + [TOKEN_SEP])

        for paragraph in document['content']:
            chars.extend([ch for ch in paragraph])
            tokens.extend(tokenizer.tokenize(paragraph))

        chars = ' '.join(chars)
        record = {
            'content-key': document['content-key'],
            'title': document['title'],
            'chars': chars,
            'tokens': tokens
        }

        es.index(index=es_index, doc_type=es_doc_type, body=record)


def create_paragraph_index(es_host, es_index, es_doc_type):
    es = Elasticsearch([{'host': es_host, 'port': 9200}])

    mapping = {
        "mappings": {
            es_doc_type: {
                "properties": {
                    "content-key": {
                        "type": "keyword"
                    },
                    "detail": {
                        "type": "keyword"
                    },
                    "chars": {
                        "type": "text",
                        "analyzer": "ik_smart",
                        "fields": {
                            "cn": {
                                "type": "text",
                                "analyzer": "ik_smart"
                            },
                            "en": {
                                "type": "text",
                                "analyzer": "english"
                            }
                        }
                    },
                    "bigrams": {
                        "type": "text",
                        "analyzer": "ik_smart",
                        "fields": {
                            "cn": {
                                "type": "text",
                                "analyzer": "ik_smart"
                            },
                            "en": {
                                "type": "text",
                                "analyzer": "english"
                            }
                        }
                    },
                    "words": {
                        "type": "text",
                        "analyzer": "ik_smart",
                        "fields": {
                            "cn": {
                                "type": "text",
                                "analyzer": "ik_smart"
                            },
                            "en": {
                                "type": "text",
                                "analyzer": "english"
                            }
                        }
                    }
                }
            }
        }
    }

    if es.indices.exists(index=es_index):
        logging.info('index %s already exists' % es_index)
    else:
        result = es.indices.create(index=es_index, body=mapping)
        logging.info('create index %s' % es_index)
        logging.info(result)


def insert_paragraph_index(es_host, es_index, es_doc_type, data_path, tokenizer=None, with_title=True):
    es = Elasticsearch([{'host': es_host, 'port': 9200}])
    logging.info('create es index %s' % es_index)
    create_paragraph_index(es_host, es_index, es_doc_type)

    paragraphs = get_paragraphs(data_path, tokenizer, with_title=with_title)
    for para in tqdm(paragraphs):
        es.index(index=es_index, doc_type=es_doc_type, body=para)


def create_dpr_from_es(data_path, es_host, es_port, es_index, es_doc_type, es_retrieve_size, es_retrieve_token,
                       tokenizer, with_title=True):
    logging.info('reading data from %s' % data_path)
    documents = get_documents_with_paragraphs('%s/content.xlsx' % data_path, tokenizer, with_title=with_title)

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
                if len(line['answer']) == 0:
                    continue

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

                question = [tok for tok in str(line['question'])]
                question = ' '.join(question)
                retrieve_paragraphs = elasticsearch_utils.retrieve(es_host, es_port, es_index, es_doc_type,
                                                                   es_retrieve_size, es_retrieve_token, question)
                for para in retrieve_paragraphs:
                    selected = False
                    para = para['_source']
                    for answer in paragraphs:
                        if (para['content-key'] == answer['content-key']) and (para['detail'] == answer['detail']):
                            selected = True
                            break
                    if selected:
                        continue
                    else:
                        paragraphs.append({
                            'content-key': para['content-key'],
                            'detail': para['detail'],
                            'chars': para['chars'].split(),
                            'tokens': para['tokens'],
                            'selected': False,
                        })
                        break

                records.append({
                    'question': line['question'],
                    'aim': line['aim'],
                    'paragraphs': paragraphs
                })

        logging.info('saving %s data' % split)
        with codecs.open('%s/%s_dpr.txt' % (data_path, split), 'w', 'utf-8') as fout:
            for record in tqdm(records):
                fout.write('%s\n' % (json.dumps(record)))

    logging.info('complete create dpr data ')


def create_dpr_from_faiss(data_path, output_path, faiss_path, faiss_state, tokenizer, model,
                          max_question_len, negative_size, with_title=True, use_cpu=False):
    documents = get_documents_with_paragraphs('%s/content.xlsx' % data_path, tokenizer, with_title=with_title)

    logging.info('loading faiss index')
    faiss_id_path = '%s/faiss_%s.id' % (faiss_path, faiss_state)
    faiss_data_path = '%s/faiss_%s.data' % (faiss_path, faiss_state)
    faiss_idx_to_id = faiss_utils.load_faiss_id(faiss_id_path)
    faiss_index = faiss.read_index(faiss_data_path)
    gpu_res = faiss.StandardGpuResources()
    faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss_index)

    data_splits = ['train', 'valid']
    with torch.no_grad():
        for split in data_splits:
            logging.info('processing %s data' % split)

            records = []
            with codecs.open('%s/%s.txt' % (data_path, split), 'r', 'utf-8') as fin:
                for line in tqdm(fin):
                    line = line.strip()
                    if line == '':
                        continue

                    line = json.loads(line)
                    if len(line['answer']) == 0:
                        continue

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

                    retrieve_paras, _ = faiss_utils.retrieve_from_faiss(tokenizer, model,
                                                                        faiss_idx_to_id, faiss_index,
                                                                        line['question'],
                                                                        retrieve_size=negative_size * 2,
                                                                        max_question_len=max_question_len,
                                                                        use_cpu=use_cpu)

                    add_negative_size = 0
                    for retrieve_para in retrieve_paras:
                        if add_negative_size >= negative_size:
                            break

                        retrieve_para = json.loads(retrieve_para)
                        selected = False
                        for answer in paragraphs:
                            if (retrieve_para['content-key'] == answer['content-key']) and (
                                    retrieve_para['detail'] == answer['detail']):
                                selected = True
                                break
                        if not selected:
                            document = documents[retrieve_para['content-key']]
                            for para in document:
                                if para['detail'] == retrieve_para['detail']:
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
            with codecs.open('%s/%s_dpr.txt' % (output_path, split), 'w', 'utf-8') as fout:
                for record in tqdm(records):
                    fout.write('%s\n' % (json.dumps(record)))

    logging.info('complete create dpr data ')


def statistics_paragraphs(datapath):
    context_len_dict = {128: 0, 256: 0, 384: 0, 400: 0, 512: 0, 768: 0, 1024: 0, 2048: 0, 4096: 0}
    question_len_dict = {16: 0, 24: 0, 32: 0, 64: 0, 128: 0}

    data_split = ['train', 'valid']
    for ds in data_split:
        logging.info('%s data count' % ds)
        with codecs.open('%s/%s_dpr.txt' % (datapath, ds), 'r', 'utf-8') as fin:
            for line in fin:
                line = json.loads(line)

                que_len = len(str(line['question']))
                for ql in question_len_dict:
                    if que_len <= ql:
                        question_len_dict[ql] = question_len_dict[ql] + 1
                        break

                paras = line['search_paras']
                for para in paras:
                    context_len = len(para['content'])
                    for cl in context_len_dict:
                        if context_len <= cl:
                            context_len_dict[cl] = context_len_dict[cl] + 1
                            break

        logging.info('context length: %s' % str(context_len_dict))
        logging.info('question length: %s' % str(question_len_dict))

        for cl in context_len_dict:
            context_len_dict[cl] = 0
        for ql in question_len_dict:
            question_len_dict[ql] = 0


class DprDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_question_len=0, max_context_len=0,
                 with_sep=False, debug=False):
        super(DprDataset, self).__init__()
        self.data = []

        with codecs.open(data_path, 'r', 'utf-8') as fin:
            question_id = 0
            for line in fin:
                line = json.loads(line)

                question_id += 1
                question = text_utils.dbc_to_sbc(str(line['question']))
                # question = tokenizer.tokenize(question)
                question = [ch for ch in question]
                question = [TOKEN_CLS] + question
                if max_question_len > 0:
                    question = question[:max_question_len]

                question_tokens = tokenizer.convert_tokens_to_ids(question)
                question_tokens_type_id = [0] * len(question_tokens)

                positives = []
                negatives = []
                for para in line['paragraphs']:
                    selected = para['selected']

                    paragraph_tokens = [TOKEN_CLS] + para['chars']
                    if not with_sep:
                        while TOKEN_SEP in paragraph_tokens:
                            paragraph_tokens.remove(TOKEN_SEP)
                    if max_context_len > 0:
                        paragraph_tokens = paragraph_tokens[:max_context_len]
                    paragraph_tokens = tokenizer.convert_tokens_to_ids(paragraph_tokens)
                    paragraph_tokens_type_id = [0] * len(paragraph_tokens)

                    if selected:
                        positives.append([paragraph_tokens, paragraph_tokens_type_id])
                    else:
                        negatives.append([paragraph_tokens, paragraph_tokens_type_id])

                for pos_idx in range(len(positives)):
                    for neg_idx in range(len(negatives)):
                        pair_list = []
                        pair_list.append([positives[pos_idx][0], positives[pos_idx][1], 1])
                        pair_list.append([negatives[neg_idx][0], negatives[neg_idx][1], 0])

                        self.data.append([question_id, question_tokens, question_tokens_type_id, pair_list])

                if debug and (len(self.data) >= 100):
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# if __name__ == '__main__':
#     from transformers import BertTokenizer
#
#     logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
#     logging.getLogger('elasticsearch').setLevel(logging.ERROR)
#
#     data_path = '../data/dataset'
#     doc_file = 'content.xlsx'
#     pretrained_bert_path = '../data/bert/bert-base-chinese/'
#
#     es_host = '10.79.169.16'
#     es_port = 9200
#     es_index_doc = 'air-travel-cup-document'
#     es_index_para = 'air-travel-cup-paragraph'
#     es_doc_type = 'record'
#     es_retrieve_size = 10
#     es_retrieve_token = 'chars'
#
#     # tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % pretrained_bert_path)
#
#     # insert_document_index(es_host, es_index_doc, es_doc_type, '%s/%s' % (data_path, doc_file),
#     #                       tokenizer, with_title=True)
#     # insert_paragraph_index(es_host, es_index_para, es_doc_type, '%s/%s' % (data_path, doc_file),
#     #                        tokenizer=tokenizer, with_title=True)
#
#     # create_dpr_from_es(data_path, es_host, es_port, es_index_para, es_doc_type, es_retrieve_size, es_retrieve_token,
#     #                    tokenizer, with_title=True)
#
#     # statistics_paragraphs(data_path)
