import os
import math
import json
import codecs
import logging
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import faiss
from torch.nn.utils import rnn
from transformers import BertTokenizer

from reader import evaluator as reader_evaluator
from utils import torch_utils, text_utils, faiss_utils
from utils.data_utils import read_document, get_documents_with_paragraphs, TOKEN_CLS, TOKEN_SEP


def init(args):
    logging.info('loading embedding')
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info('loading documents')
    documents = read_document('%s/content.xlsx' % args.data_path)
    documents = {record['content-key']: record for record in documents}
    documents_with_title = get_documents_with_paragraphs('%s/content.xlsx' % args.data_path, tokenizer, with_title=True)

    logging.info('loading faiss index: %s' % args.faiss_path)
    faiss_idx_to_id = faiss_utils.load_faiss_id('%s.id' % args.faiss_path)
    faiss_index = faiss.read_index('%s.data' % args.faiss_path)
    gpu_res = faiss.StandardGpuResources()
    faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss_index)

    logging.info('loading retriever model: %s' % args.retriever_model_path)
    retriever_model, _, _, _ = torch_utils.load(args.retriever_model_path)
    retriever_model = retriever_model.cpu() if args.use_cpu else retriever_model.cuda()
    retriever_model.eval()

    ranker_model = None
    if args.use_ranker:
        logging.info('loading ranker model: %s' % args.ranker_model_path)
        ranker_model, _, _, _ = torch_utils.load(args.ranker_model_path)
        ranker_model = ranker_model.cpu() if args.use_cpu else ranker_model.cuda()
        ranker_model.eval()

    reranker_model = None
    if args.use_reranker:
        logging.info('loading reranker model: %s' % args.reranker_model_path)
        reranker_model, _, _, _ = torch_utils.load(args.reranker_model_path)
        reranker_model = reranker_model.cpu() if args.use_cpu else reranker_model.cuda()
        reranker_model.eval()

    logging.info('loading validator model: %s' % args.validator_model_path)
    validator_model, _, _, _ = torch_utils.load(args.validator_model_path)
    validator_model = validator_model.cpu() if args.use_cpu else validator_model.cuda()
    validator_model.eval()

    logging.info('loading reader model: %s' % args.reader_model_path)
    reader_model, _, _, _ = torch_utils.load(args.reader_model_path)
    reader_model = reader_model.cpu() if args.use_cpu else reader_model.cuda()
    reader_model.eval()

    return tokenizer, documents, documents_with_title, faiss_idx_to_id, faiss_index, \
           retriever_model, ranker_model, reranker_model, validator_model, reader_model


def get_retrieve_scores(args, documents, tokenizer, ranker_model, reranker_model, question, retrieve_ids):
    if args.use_ranker:
        paragraphs = []
        for retrieve_id in retrieve_ids:
            document = documents[retrieve_id['content-key']]
            for para in document:
                if retrieve_id['detail'] == para['detail']:
                    content = ''.join(para['chars'].split())
                    paragraphs.append(content)

        tokens_list = []
        tokens_type_id_list = []
        text_list = []

        question = text_utils.dbc_to_sbc(str(question))
        question = [ch for ch in question]
        if args.max_question_len > 0:
            question = question[:args.max_question_len]
        question = [TOKEN_CLS] + question + [TOKEN_SEP]
        question_id = tokenizer.convert_tokens_to_ids(question)

        for para in paragraphs:
            para = text_utils.dbc_to_sbc(str(para))
            para = [ch for ch in para]

            if args.max_context_len > 0:
                para = para[:(args.max_context_len - len(question))]

            tokens = question_id + tokenizer.convert_tokens_to_ids(para)
            tokens_type_id = [0] * len(question_id) + [1] * len(para)
            text = question + para

            tokens_list.append(tokens)
            tokens_type_id_list.append(tokens_type_id)
            text_list.append(text)

        tokens_list = [torch.LongTensor(np.array(item)) for item in tokens_list]
        tokens_type_id_list = [torch.LongTensor(np.array(item)) for item in tokens_type_id_list]

        tokens_list = [item.cpu() if args.use_cpu else item.cuda() for item in tokens_list]
        tokens_type_id_list = [item.cpu() if args.use_cpu else item.cuda() for item in tokens_type_id_list]

        # embeddings = []
        rank_logits = []
        batch_num = math.ceil(len(tokens_list) / args.batch_size)
        for batch_idx in range(batch_num):
            batch_end = args.batch_size * (batch_idx + 1)
            batch_end = batch_end if batch_end <= len(tokens_list) else len(tokens_list)
            b_tokens = tokens_list[args.batch_size * batch_idx: batch_end]
            b_tokens_type_id = tokens_type_id_list[args.batch_size * batch_idx: batch_end]

            b_tokens = rnn.pad_sequence(b_tokens, batch_first=True)
            b_tokens_type_id = rnn.pad_sequence(b_tokens_type_id, batch_first=True)

            b_tokens_mask = (b_tokens > 0).int()
            b_tokens_mask = b_tokens_mask.cpu() if args.use_cpu else b_tokens_mask.cuda()

            logits = ranker_model.get_score(b_tokens, b_tokens_type_id, b_tokens_mask)
            logits = logits.squeeze(-1).cpu().numpy()

            # embeddings.append(embeds)
            rank_logits.extend(logits)

        if args.use_reranker:
            embeddings = torch.cat(embeddings, dim=0)
            tokens_idx = torch.LongTensor(np.array(list(range(len(embeddings)))))

            embeddings = embeddings.unsqueeze(0)
            tokens_idx = tokens_idx.unsqueeze(0)
            embeddings = embeddings.cpu() if args.use_cpu else embeddings.cuda()
            tokens_idx = tokens_idx.cpu() if args.use_cpu else tokens_idx.cuda()

            rerank_logits = reranker_model(embeddings, tokens_idx)
            rerank_logits = rerank_logits.squeeze(0).cpu().numpy()

            scores = rerank_logits
        else:
            scores = rank_logits
    else:
        scores = list(range(len(retrieve_ids), 0, -1))

    return scores


def get_validate_ids(args, documents, tokenizer, validator_model, question, retrieve_ids, retrieve_scores):
    scores_sort = np.argsort(retrieve_scores)
    scores_sort = scores_sort[::-1][0:args.accept_retrieve_size]

    sorted_ids = []
    for idx in scores_sort:
        sorted_ids.append(retrieve_ids[idx])

    paragraphs = []
    for retrieve_id in sorted_ids:
        document = documents[retrieve_id['content-key']]
        for para in document:
            if retrieve_id['detail'] == para['detail']:
                paragraphs.append(para['tokens'])

    tokens = []
    tokens_type_id = []

    question = text_utils.dbc_to_sbc(str(question))
    question = tokenizer.tokenize(question)
    question = [TOKEN_CLS] + question + [TOKEN_SEP]

    question_tokens = tokenizer.convert_tokens_to_ids(question)
    if args.max_question_len > 0:
        question_tokens = question_tokens[:args.max_question_len]
    question_tokens_type_id = [0] * len(question_tokens)

    for para in paragraphs:
        para_tokens = tokenizer.convert_tokens_to_ids(para)

        input_tokens = question_tokens + para_tokens
        input_tokens_type_id = question_tokens_type_id + [1] * len(para_tokens)
        if args.max_context_len > 0:
            input_tokens = input_tokens[:args.max_context_len]
            input_tokens_type_id = input_tokens_type_id[:args.max_context_len]

        tokens.append(torch.LongTensor(input_tokens))
        tokens_type_id.append(torch.LongTensor(input_tokens_type_id))

    tokens = rnn.pad_sequence(tokens, batch_first=True)
    tokens_type_id = rnn.pad_sequence(tokens_type_id, batch_first=True)
    tokens_mask = (tokens > 0).int()

    tokens = tokens.cpu() if args.use_cpu else tokens.cuda()
    tokens_type_id = tokens_type_id.cpu() if args.use_cpu else tokens_type_id.cuda()
    tokens_mask = tokens_mask.cpu() if args.use_cpu else tokens_mask.cuda()

    scores = validator_model(tokens, tokens_type_id, tokens_mask)
    preds = torch.argmax(scores, dim=1)
    preds = preds.cpu().numpy()

    validate_ids = []
    for idx in range(len(preds)):
        if preds[idx] == 1:
            validate_ids.append(sorted_ids[idx])

    return validate_ids


def get_retrieve_and_validate_ids(args, documents_with_title, faiss_idx_to_id, faiss_index, tokenizer,
                                  retriever_model, ranker_model, reranker_model, validator_model, question):
    retrieve_ids, _ = faiss_utils.retrieve_from_faiss(tokenizer, retriever_model,
                                                      faiss_idx_to_id, faiss_index, question,
                                                      args.retrieve_size, args.max_question_len)
    retrieve_ids = [json.loads(obj) for obj in retrieve_ids]

    retrieve_scores = get_retrieve_scores(args, documents_with_title, tokenizer,
                                          ranker_model, reranker_model, question, retrieve_ids)
    validate_ids = get_validate_ids(args, documents_with_title, tokenizer, validator_model,
                                    question, retrieve_ids, retrieve_scores)

    return retrieve_ids, retrieve_scores, validate_ids


def get_document_result(args, retrieve_ids, retrieve_scores, answers=None):
    scores_sort = np.argsort(retrieve_scores)
    scores_sort = scores_sort[::-1][0:1000]

    sorted_ids = set()
    for idx in scores_sort:
        if len(sorted_ids) >= args.accept_retrieve_size:
            break
        sorted_ids.add(retrieve_ids[idx]['content-key'])
    sorted_ids = list(sorted_ids)

    result = 0
    if answers is not None:
        if len(answers) == 0:
            result = 1
        else:
            for record in sorted_ids:
                for item in answers:
                    if record == item['content-key']:
                        result = 1
                        break
    return sorted_ids, result


def get_paragraph_result(args, retrieve_ids, para_scores, answers=None):
    scores_sort = np.argsort(para_scores)
    scores_sort = scores_sort[::-1][0:args.accept_retrieve_size]

    sorted_ids = []
    for idx in scores_sort:
        sorted_ids.append(retrieve_ids[idx])

    result = 0
    if answers is not None:
        if len(answers) == 0:
            result = 1
        else:
            for record in sorted_ids:
                for item in answers:
                    if record['content-key'] == item['content-key'] and record['detail'] == item['detail']:
                        result = 1
                        break
    return sorted_ids, result


def get_reader_answer(args, documents, tokenizer, reader_model, question, validate_ids, answers=None):
    pred_answers = []
    pred_answers_info = []

    if len(validate_ids) > 0:
        paragraphs = []
        for validate_id in validate_ids:
            document = documents[validate_id['content-key']]
            content = document['content_label']
            for detail_idx in validate_id['detail']:
                content = content[detail_idx]
            paragraphs.append(content)

        tokens = []
        tokens_type_id = []
        texts = []

        question = text_utils.dbc_to_sbc(str(question))
        question = tokenizer.tokenize(question)
        question = [TOKEN_CLS] + question + [TOKEN_SEP]

        question_tokens = tokenizer.convert_tokens_to_ids(question)
        if args.max_question_len > 0:
            question_tokens = question_tokens[:args.max_question_len]
        question_tokens_type_id = [0] * len(question_tokens)

        for para in paragraphs:
            para = text_utils.dbc_to_sbc(str(para))
            para = tokenizer.tokenize(para)

            if args.max_context_len > 0:
                para = para[:(args.max_context_len - len(question_tokens))]

            input_tokens = question_tokens + tokenizer.convert_tokens_to_ids(para)
            input_tokens_type_id = question_tokens_type_id + [1] * len(para)
            text = question + para

            tokens.append(torch.LongTensor(input_tokens))
            tokens_type_id.append(torch.LongTensor(input_tokens_type_id))
            texts.append(text)

        tokens = rnn.pad_sequence(tokens, batch_first=True)
        tokens_type_id = rnn.pad_sequence(tokens_type_id, batch_first=True)
        tokens_mask = (tokens > 0).int()

        tokens = tokens.cpu() if args.use_cpu else tokens.cuda()
        tokens_type_id = tokens_type_id.cpu() if args.use_cpu else tokens_type_id.cuda()
        tokens_mask = tokens_mask.cpu() if args.use_cpu else tokens_mask.cuda()

        starts, ends = reader_model(tokens, tokens_type_id, tokens_mask)
        starts = torch.argmax(starts, dim=1)
        ends = torch.argmax(ends, dim=1)
        starts = starts.cpu().numpy()
        ends = ends.cpu().numpy()

        detail_type = 'y-0' if len(starts) > 1 else 'n'
        for idx in range(len(validate_ids)):
            start = starts[idx]
            end = ends[idx]
            text = texts[idx]

            if end >= start:
                pred_answers.append(''.join(text[start: end + 1]))
                pred_answers_info.append({
                    'content-key': validate_ids[idx]['content-key'],
                    'detail': validate_ids[idx]['detail'],
                    'location': [int(start - len(question_tokens)), int(end - len(question_tokens))],
                    'detail-type': detail_type
                })

    if len(pred_answers) == 0:
        pred_answers.append('')

    reader_f1 = 0
    reader_em = 0
    if answers is not None:
        gold_answers = []
        if len(answers) == 0:
            gold_answers.append('')
        else:
            for answer in answers:
                document = documents[answer['content-key']]
                content = document['content_label']
                for detail_idx in answer['detail']:
                    content = content[detail_idx]
                gold_answers.append(content[answer['location'][0]: answer['location'][1] + 1])

        f1_list = []
        em_list = []
        for answer in pred_answers:
            f1_list.append(reader_evaluator.calc_f1_score(gold_answers, answer))
            em_list.append(reader_evaluator.calc_em_score(gold_answers, answer))

        reader_f1 = max(f1_list)
        reader_em = max(em_list)
    return pred_answers_info, reader_f1, reader_em


def get_pred_result(args, documents, tokenizer, reader_model, question,
                    retrieve_ids, retrieve_scores, validate_ids, answers=None):
    docs_id, doc_success = get_document_result(args, retrieve_ids, retrieve_scores, answers=answers)
    paras_id, para_success = get_paragraph_result(args, retrieve_ids, retrieve_scores, answers=answers)
    reader_answers, reader_f1, reader_em = get_reader_answer(args, documents, tokenizer, reader_model,
                                                             question, validate_ids, answers=answers)

    return docs_id, paras_id, reader_answers, doc_success, para_success, reader_f1, reader_em


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    torch_utils.setup_seed(0)

    output_path = '%s/faiss' % args.output_path
    if args.use_ranker:
        output_path += '_rank'
    if args.use_reranker:
        output_path += '_rerank'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    tokenizer, documents, documents_with_title, faiss_idx_to_id, faiss_index, \
    retriever_model, ranker_model, reranker_model, validator_model, reader_model = init(args)

    with codecs.open('%s/eval.log' % output_path, 'w', 'utf-8') as fout:
        logging.info('========== config ==========')
        logging.info('use ranker: %s' % args.use_ranker)
        logging.info('use reranker: %s' % args.use_reranker)
        fout.write('========== config ==========\n')
        fout.write('use ranker: %s\n' % args.use_ranker)
        fout.write('use reranker: %s\n' % args.use_reranker)

        with torch.no_grad():
            data_splits = ['train', 'valid']
            for split in data_splits:
                logging.info('========== %s data ==========' % split)
                fout.write('========== %s data ==========\n' % split)

                total_num = 0
                doc_success_num = 0
                para_success_num = 0
                reader_f1_sum = 0
                reader_em_sum = 0
                with codecs.open('%s/%s.txt' % (args.data_path, split), 'r', 'utf-8') as fin:
                    for line in tqdm(fin):
                        total_num += 1
                        if args.debug and total_num > 10:
                            break

                        line = line.strip()
                        if line == '':
                            continue

                        line = json.loads(line)
                        question = str(line['question'])
                        answers = line['answer']

                        retrieve_ids, retrieve_scores, validate_ids = get_retrieve_and_validate_ids(
                            args, documents_with_title, faiss_idx_to_id, faiss_index, tokenizer,
                            retriever_model, ranker_model, reranker_model, validator_model, question)

                        _, _, _, doc_success, para_success, reader_f1, reader_em = get_pred_result(
                            args, documents, tokenizer, reader_model, question,
                            retrieve_ids, retrieve_scores, validate_ids, answers=answers)

                        doc_success_num += doc_success
                        para_success_num += para_success
                        reader_f1_sum += reader_f1
                        reader_em_sum += reader_em

                doc_recall = doc_success_num / total_num
                para_recall = para_success_num / total_num
                reader_f1 = reader_f1_sum / total_num
                reader_em = reader_em_sum / total_num

                logging.info('doc recall: %.3f' % doc_recall)
                logging.info('paragraph recall: %.3f' % para_recall)
                logging.info('reader f1: %.3f, em: %.3f' % (reader_f1, reader_em))
                fout.write('doc recall: %.3f\n' % doc_recall)
                fout.write('paragraph recall: %.3f\n' % para_recall)
                fout.write('reader f1: %.3f, em: %.3f\n' % (reader_f1, reader_em))

    logging.info('complete testing')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='../data/dataset/')
    parser.add_argument('--output_path', type=str,
                        default='../runtime/pipline_tok_sep')
    parser.add_argument('--pretrained_bert_path', type=str,
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--retriever_model_path', type=str,
                        default='../runtime/retriever_tok_sep/dpr_ance/last.pth')
    parser.add_argument('--faiss_path', type=str,
                        default='../runtime/retriever_tok_sep/dpr_ance/faiss_last')
    parser.add_argument('--ranker_model_path', type=str,
                        default='../runtime/ranker_tok_sep/finetune/last.pth')
    parser.add_argument('--reranker_model_path', type=str,
                        default='../runtime/reranker/last.pth')
    parser.add_argument('--validator_model_path', type=str,
                        default='../runtime/validator_sep/best.pth')
    parser.add_argument('--reader_model_path', type=str,
                        default='../runtime/reader_tok/best.pth')
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--retrieve_size', type=int,
                        default=1000)
    parser.add_argument('--batch_size', type=int,
                        default=64)
    parser.add_argument('--use_ranker', type=bool,
                        default=True)
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
