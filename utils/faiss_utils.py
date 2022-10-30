import json
import codecs
import logging
import numpy as np
from tqdm import tqdm
import faiss
import torch
from transformers import BertTokenizer

from utils import text_utils, torch_utils
from utils.data_utils import get_paragraphs, TOKEN_CLS, TOKEN_SEP


def load_faiss_id(data_path):
    idx_to_id = {}
    with codecs.open(data_path, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line == '':
                continue

            idx_to_id[len(idx_to_id)] = line
    return idx_to_id


def retrieve_from_faiss(tokenizer, model, faiss_idx_to_id, faiss_index, question,
                        retrieve_size=10, max_question_len=0, use_cpu=False):
    question = text_utils.dbc_to_sbc(str(question))
    # question = tokenizer.tokenize(question)
    question = [ch for ch in question]
    if max_question_len > 0:
        question = question[:max_question_len]

    question = [TOKEN_CLS] + question
    question = tokenizer.convert_tokens_to_ids(question)
    question_segment = [0] * len(question)

    question = torch.LongTensor(question).unsqueeze(0)
    question_segment = torch.LongTensor(question_segment).unsqueeze(0)
    question = question.cpu() if use_cpu else question.cuda()
    question_segment = question_segment.cpu() if use_cpu else question_segment.cuda()

    embedding = model.get_question_embedding(question, question_segment)
    embedding = embedding.detach().cpu().numpy()

    faiss_sco, faiss_idx = faiss_index.search(embedding, retrieve_size)

    retrieve_ids = [faiss_idx_to_id[idx] for idx in faiss_idx[0]]
    faiss_sco = faiss_sco[0]
    return retrieve_ids, faiss_sco


def build_embedding(data_path, output_path, model_state, tokenizer, model, max_context_len,
                    with_title=True, with_sep=True, use_cpu=False):
    paragraphs = get_paragraphs('%s/content.xlsx' % data_path, tokenizer, with_title=with_title)

    output_path = '%s/%s.emb' % (output_path, model_state)
    logging.info('Writing to %s' % output_path)

    model.eval()
    with torch.no_grad():
        with codecs.open(output_path, 'w', 'utf-8') as fout:
            for para in tqdm(paragraphs):
                para_id = {'content-key': para['content-key'], 'detail': para['detail']}
                para_id = json.dumps(para_id)

                # paragraph_tokens = para['tokens']
                paragraph_tokens = para['chars'].split()
                if not with_sep:
                    while TOKEN_SEP in paragraph_tokens:
                        paragraph_tokens.remove(TOKEN_SEP)

                paragraph_tokens = [TOKEN_CLS] + paragraph_tokens
                if max_context_len > 0:
                    paragraph_tokens = paragraph_tokens[:max_context_len]
                paragraph_tokens = tokenizer.convert_tokens_to_ids(paragraph_tokens)
                paragraph_tokens_type_id = [0] * len(paragraph_tokens)

                paragraph_tokens = torch.LongTensor(paragraph_tokens).unsqueeze(0)
                paragraph_tokens_type_id = torch.LongTensor(paragraph_tokens_type_id).unsqueeze(0)
                paragraph_tokens = paragraph_tokens.cpu() if use_cpu else paragraph_tokens.cuda()
                paragraph_tokens_type_id = paragraph_tokens_type_id.cpu() if use_cpu else paragraph_tokens_type_id.cuda()

                embedding = model.get_context_embedding(paragraph_tokens, paragraph_tokens_type_id)
                embedding = embedding.squeeze(0).detach().cpu().numpy()

                embedding = [str(emb) for emb in embedding]
                embedding = ','.join(embedding)

                fout.write('%s\t%s\n' % (para_id, embedding))


def build_faiss_index(output_path, model_state, batch_size=4096, debug=False):
    ids = []
    embed_path = '%s/%s.emb' % (output_path, model_state)
    logging.info('Reading %s' % embed_path)
    with codecs.open(embed_path, 'r', 'utf-8') as fin:
        batch_data = []

        line = fin.readline()
        id, embed = line.split('\t')
        embed = [float(e) for e in embed.split(',')]

        ids.append(id)
        batch_data.append(embed)
        embed_size = len(embed)

        faiss_index = faiss.IndexFlatIP(embed_size)
        for line in tqdm(fin):
            id, embed = line.split('\t')
            embed = [float(e) for e in embed.split(',')]

            ids.append(id)
            batch_data.append(embed)

            if len(batch_data) % batch_size == 0:
                batch_data = np.array(batch_data).astype('float32')
                faiss_index.add(batch_data)
                batch_data = []

            if debug and (faiss_index.ntotal >= 2000):
                break

        if len(batch_data) > 0:
            batch_data = np.array(batch_data).astype('float32')
            faiss_index.add(batch_data)

        assert len(ids) == faiss_index.ntotal
        logging.info('Total index num %s' % faiss_index.ntotal)

    faiss_id_path = '%s/faiss_%s.id' % (output_path, model_state)
    logging.info('Writing faiss id to %s' % faiss_id_path)
    with codecs.open(faiss_id_path, 'w', 'utf-8') as fout:
        for id in ids:
            fout.write('%s\n' % id)

    faiss_data_path = '%s/faiss_%s.data' % (output_path, model_state)
    logging.info('Writing faiss data to %s' % faiss_data_path)
    faiss.write_index(faiss_index, faiss_data_path)


def build_faiss_data(data_path, output_path, model_state, tokenizer, model,
                     max_context_len, batch_size=4096, with_title=True, with_sep=True, use_cpu=False, debug=False):
    logging.info('Begin build embedding')
    build_embedding(data_path, output_path, model_state, tokenizer, model, max_context_len,
                    with_title=with_title, with_sep=with_sep, use_cpu=use_cpu)

    logging.info('Begin build faiss index')
    build_faiss_index(output_path, model_state, batch_size, debug=debug)


# if __name__ == '__main__':
#     logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
#
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
#     data_path = '../data/dataset/'
#     pretrained_bert_path = '../data/bert/bert-base-chinese/'
#     model_path = '../runtime/retriever/dpr_ance/'
#     model_state = 'last'
#     output_path = '../runtime/retriever/dpr_ance/'
#     max_context_len = 512
#     batch_size = 4096
#     with_title = True
#     with_sep = False
#     use_cpu = False
#     debug = False
#
#     logging.info('Begin build embedding')
#     tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % pretrained_bert_path)
#
#     logging.info('loading model: %s' % model_path)
#     model_path = '%s/%s.pth' % (model_path, model_state)
#     model, _, _, _ = torch_utils.load(model_path)
#     model = model.cpu() if use_cpu else model.cuda()
#
#     build_faiss_data(data_path, output_path, model_state, tokenizer, model,
#                      max_context_len=max_context_len, batch_size=batch_size,
#                      with_title=with_title, with_sep=with_sep, use_cpu=use_cpu, debug=debug)
