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

        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, tokens_type_id, tokens_mask):
        out = self.embedding(input_ids=tokens, token_type_ids=tokens_type_id, attention_mask=tokens_mask)
        out = out.last_hidden_state
        out = out[:, 0, :]
        logits = self.linear(out)
        return logits
