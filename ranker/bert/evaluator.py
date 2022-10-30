import math
import json
import codecs
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.utils import rnn

from utils import text_utils
from utils.data_utils import TOKEN_CLS, TOKEN_SEP


def check_recall(labels, logits, accept_size):
    logits_sorted = np.argsort(logits)
    logits_sorted = logits_sorted[::-1][0:accept_size]

    success = 0
    for pred in logits_sorted:
        if pred in labels:
            success = 1
            break

    return success


def check_question(tokenizer, model, question, paragraphs, accept_size,
                   max_question_len, max_context_len, batch_size, with_sep=False, use_cpu=False):
    question = text_utils.dbc_to_sbc(str(question))
    # question = tokenizer.tokenize(question)
    question = [ch for ch in question]
    question = [TOKEN_CLS] + question + [TOKEN_SEP]

    question_tokens = tokenizer.convert_tokens_to_ids(question)
    if max_question_len > 0:
        question_tokens = question_tokens[:max_question_len]
    question_tokens_type_id = [0] * len(question_tokens)

    logits = []
    batch_num = math.ceil(len(paragraphs) / batch_size)
    for batch_idx in range(batch_num):
        batch_start = batch_size * batch_idx
        batch_end = batch_size * (batch_idx + 1)
        batch_end = batch_end if batch_end <= len(paragraphs) else len(paragraphs)
        batch_paras = paragraphs[batch_start: batch_end]

        b_tokens = []
        b_tokens_type_id = []
        for para in batch_paras:
            para_tokens = para['chars']
            if not with_sep:
                while TOKEN_SEP in para_tokens:
                    para_tokens.remove(TOKEN_SEP)
            para_tokens = tokenizer.convert_tokens_to_ids(para_tokens)
            para_tokens_type_id = [1] * len(para_tokens)

            if max_context_len > 0:
                para_tokens = para_tokens[:max_context_len - len(question)]
                para_tokens_type_id = para_tokens_type_id[:max_context_len - len(question)]

            para_tokens = question_tokens + para_tokens
            para_tokens_type_id = question_tokens_type_id + para_tokens_type_id
            b_tokens.append(torch.LongTensor(para_tokens))
            b_tokens_type_id.append(torch.LongTensor(para_tokens_type_id))

        b_tokens = rnn.pad_sequence(b_tokens, batch_first=True)
        b_tokens_type_id = rnn.pad_sequence(b_tokens_type_id, batch_first=True)
        b_tokens_mask = (b_tokens > 0).int()

        b_tokens = b_tokens.cpu() if use_cpu else b_tokens.cuda()
        b_tokens_type_id = b_tokens_type_id.cpu() if use_cpu else b_tokens_type_id.cuda()
        b_tokens_mask = b_tokens_mask.cpu() if use_cpu else b_tokens_mask.cuda()

        b_logits = model.get_score(b_tokens, b_tokens_type_id, b_tokens_mask)
        b_logits = b_logits.squeeze(-1).cpu().numpy()
        logits.extend(b_logits)

    labels = np.array([int(para['selected']) for para in paragraphs])
    labels = list(np.where(labels == 1)[0])

    logits_top_5 = logits[:5]
    logits_top_10 = logits[:10]
    logits_top_50 = logits[:50]
    logits_top_100 = logits[:100]
    top_5_success = check_recall(labels, logits_top_5, accept_size)
    top_10_success = check_recall(labels, logits_top_10, accept_size)
    top_50_success = check_recall(labels, logits_top_50, accept_size)
    top_100_success = check_recall(labels, logits_top_100, accept_size)

    logits_sorted = np.argsort(logits)
    logits_sorted = logits_sorted[::-1][0:accept_size]

    return logits_sorted, [top_5_success, top_10_success, top_50_success, top_100_success]


def evaluate(data_path, tokenizer, model, accept_size, max_question_len, max_context_len, batch_size,
             with_sep=False, use_cpu=False, debug=False):
    total_num = 0
    top_5_success_num = 0
    top_10_success_num = 0
    top_50_success_num = 0
    top_100_success_num = 0
    with codecs.open(data_path, 'r', 'utf-8') as fin:
        for line in tqdm(fin):
            line = line.strip()
            if line == '':
                continue

            line = json.loads(line)
            question = line['question']
            paragraphs = line['paragraphs']

            _, [top_5_success, top_10_success, top_50_success, top_100_success] = check_question(tokenizer, model,
                                                                                                 question, paragraphs,
                                                                                                 accept_size,
                                                                                                 max_question_len,
                                                                                                 max_context_len,
                                                                                                 batch_size,
                                                                                                 with_sep=with_sep,
                                                                                                 use_cpu=use_cpu)

            total_num += 1
            top_5_success_num += top_5_success
            top_10_success_num += top_10_success
            top_50_success_num += top_50_success
            top_100_success_num += top_100_success

            if debug and total_num >= 10:
                break

    top_5_recall = top_5_success_num / total_num
    top_10_recall = top_10_success_num / total_num
    top_50_recall = top_50_success_num / total_num
    top_100_recall = top_100_success_num / total_num

    return top_5_recall, top_10_recall, top_50_recall, top_100_recall
