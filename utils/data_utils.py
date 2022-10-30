import copy
import xlrd
import json
from tqdm import tqdm

TOKEN_PAD = '[PAD]'
TOKEN_UNK = '[UNK]'
TOKEN_CLS = '[CLS]'
TOKEN_SEP = '[SEP]'
TOKEN_MASK = '[MASK]'
TABLE_SEP = ['[unused1]', '[unused2]']
IMG_SEP = ['[unused3]', '[unused4]']
COLBERT_Q_SPE = '[unused5]'
COLBERT_D_SPE = '[unused6]'


def read_document(data_path):
    docs = []
    workbook = xlrd.open_workbook(data_path)
    sheet = workbook.sheet_by_index(0)
    for row_idx in range(1, sheet.nrows):
        title = sheet.cell(row_idx, 0).value
        content = sheet.cell(row_idx, 1).value
        content_label = sheet.cell(row_idx, 2).value
        content_key = sheet.cell(row_idx, 3).value

        content = content.split('\n')
        content_label = json.loads(content_label)

        docs.append({
            'content-key': content_key,
            'title': title,
            'content': content,
            'content_label': content_label
        })

    return docs


def read_paragraph(paragraphs, infos, content_key, titles, prefix, tokenizer=None, with_title=True):
    if isinstance(infos, dict):
        for key in infos:
            if key == 'title':
                if infos[key] is not None:
                    title = infos[key].strip()
                    if title != '':
                        titles.append(infos[key])
                continue

            prefix_exp = copy.deepcopy(prefix)
            prefix_exp.append(key)
            if key in ['text', 'table', 'img']:
                chars = []
                tokens = []

                if with_title:
                    for title in titles:
                        chars.extend([ch for ch in title] + [TOKEN_SEP])
                        tokens.extend(tokenizer.tokenize(title) + [TOKEN_SEP])

                content = infos[key]
                content_chars = [ch for ch in content]
                content_tokens = tokenizer.tokenize(content)
                if key == 'table':
                    content_chars = [TABLE_SEP[0]] + content_chars + [TABLE_SEP[1]]
                    content_tokens = [TABLE_SEP[0]] + content_tokens + [TABLE_SEP[1]]
                if key == 'img':
                    content_chars = [IMG_SEP[0]] + content_chars + [IMG_SEP[1]]
                    content_tokens = [IMG_SEP[0]] + content_tokens + [IMG_SEP[1]]
                chars.extend(content_chars)
                tokens.extend(content_tokens)

                chars = ' '.join(chars)
                record = {
                    'content-key': content_key,
                    'detail': prefix_exp,
                    'chars': chars,
                    'tokens': tokens
                }
                paragraphs.append(record)
            else:
                read_paragraph(paragraphs, infos[key], content_key, titles, prefix_exp,
                               tokenizer, with_title=with_title)
    elif isinstance(infos, list):
        for idx in range(len(infos)):
            prefix_exp = copy.deepcopy(prefix)
            prefix_exp.append(idx)
            read_paragraph(paragraphs, infos[idx], content_key, titles, prefix_exp,
                           tokenizer, with_title=with_title)


def get_paragraphs(data_path, tokenizer=None, with_title=True):
    paragraphs = []
    documents = read_document(data_path)
    for document in tqdm(documents):
        title = str(document['title'])
        read_paragraph(paragraphs, document['content_label'], document['content-key'], [title], [],
                       tokenizer, with_title=with_title)
    return paragraphs


def get_documents_with_paragraphs(data_path, tokenizer, with_title=True):
    paragraphs = get_paragraphs(data_path, tokenizer, with_title=with_title)

    documents = {}
    for paragraph in paragraphs:
        record = documents.get(paragraph['content-key'], [])
        record.append(paragraph)
        documents[paragraph['content-key']] = record

    return documents
