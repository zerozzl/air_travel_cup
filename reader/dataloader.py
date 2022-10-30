import json
import codecs
from torch.utils.data import Dataset

from utils import text_utils
from utils.data_utils import TOKEN_CLS, TOKEN_SEP


class MrDataset(Dataset):
    def __init__(self, data_path, documents, tokenizer, max_context_len=0, max_question_len=0,
                 do_to_id=False, do_sort=False, debug=False):
        super(MrDataset, self).__init__()
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
                if max_question_len > 0:
                    question = question[:max_question_len]

                question_tokens = question
                if do_to_id:
                    question_tokens = tokenizer.convert_tokens_to_ids(question_tokens)
                question_tokens_type_id = [0] * len(question_tokens)

                answers = line['answer']
                for answer in answers:
                    document = documents[answer['content-key']]
                    content = document['content_label']
                    for detail_idx in answer['detail']:
                        content = content[detail_idx]
                    content = text_utils.dbc_to_sbc(content)
                    # content = tokenizer.tokenize(content)
                    content = [ch for ch in content]
                    if max_context_len > 0:
                        content = content[:(max_context_len - len(question_tokens))]

                    answer_start = answer['location'][0]
                    answer_end = answer['location'][1]
                    answer_text = content[answer_start:answer_end + 1]
                    label = [answer_start, answer_end]

                    context_len = len(content)
                    if label[1] >= context_len:
                        label[1] = context_len - 1

                    context_tokens = content
                    if do_to_id:
                        context_tokens = tokenizer.convert_tokens_to_ids(context_tokens)
                    context_tokens_type_id = [1] * len(context_tokens)

                    label = [pos + len(question_tokens) for pos in label]
                    input_tokens = question_tokens + context_tokens
                    input_tokens_type_id = question_tokens_type_id + context_tokens_type_id
                    input_text = question + content
                    answer_text = [''.join(answer_text)]

                    self.data.append([label, input_tokens, input_tokens_type_id, input_text, answer_text])

                if debug and len(self.data) >= 100:
                    break

        if do_sort:
            self.data = sorted(self.data, key=lambda x: x[3], reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
