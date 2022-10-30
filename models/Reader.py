import torch
from transformers import BertConfig, BertModel


class Bert(torch.nn.Module):
    def __init__(self, config_path, model_path, bert_freeze):
        super(Bert, self).__init__()

        config = BertConfig.from_json_file(config_path)
        self.embedding = BertModel.from_pretrained(model_path, config=config)
        self.linear = torch.nn.Linear(config.hidden_size, 2)

        if bert_freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False

    def forward(self, input_tokens, input_tokens_type_id, input_tokens_mask):
        out = self.embedding(input_ids=input_tokens,
                             token_type_ids=input_tokens_type_id,
                             attention_mask=input_tokens_mask)
        out = out.last_hidden_state
        out = self.linear(out)

        start = out[:, :, 0]
        end = out[:, :, 1]

        return start, end
