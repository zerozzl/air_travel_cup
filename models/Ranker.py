import torch
from transformers import BertConfig, BertModel


class Bert(torch.nn.Module):
    def __init__(self, config_path, model_path, bert_freeze):
        super(Bert, self).__init__()

        config = BertConfig.from_json_file(config_path)
        self.embedding = BertModel.from_pretrained(model_path, config=config)
        self.linear = torch.nn.Linear(config.hidden_size, 1)

        if bert_freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.ce_loss = torch.nn.CrossEntropyLoss()

    # def forward(self, tokens, tokens_type_id, tokens_masks):
    #     embed = self.embedding(input_ids=tokens,
    #                            token_type_ids=tokens_type_id,
    #                            attention_mask=tokens_masks)
    #     embed = embed.last_hidden_state
    #     embed = embed[:, 0, :]
    #     logits = self.linear(embed)
    #
    #     return embed, logits

    def forward(self, positive_tokens, positive_tokens_type_id, positive_tokens_mask,
                negative_tokens, negative_tokens_type_id, negative_tokens_mask):
        positive_embed = self.embedding(input_ids=positive_tokens,
                                        token_type_ids=positive_tokens_type_id,
                                        attention_mask=positive_tokens_mask)
        positive_embed = positive_embed.last_hidden_state
        positive_embed = positive_embed[:, 0, :]
        positive_logits = self.linear(positive_embed)

        negative_embed = self.embedding(input_ids=negative_tokens,
                                        token_type_ids=negative_tokens_type_id,
                                        attention_mask=negative_tokens_mask)
        negative_embed = negative_embed.last_hidden_state
        negative_embed = negative_embed[:, 0, :]
        negative_logits = self.linear(negative_embed)

        try:
            negative_logits = negative_logits.view(positive_logits.size(0), -1)
            logits = torch.cat([positive_logits, negative_logits], dim=1)
        except:
            logits = None

        return logits

    def get_score(self, tokens, tokens_type_id, tokens_mask):
        embed = self.embedding(input_ids=tokens,
                               token_type_ids=tokens_type_id,
                               attention_mask=tokens_mask)
        embed = embed.last_hidden_state
        embed = embed[:, 0, :]
        logits = self.linear(embed)
        return logits


class ColBert(torch.nn.Module):
    def __init__(self, config_path, model_path, output_size, bert_freeze):
        super(ColBert, self).__init__()

        config = BertConfig.from_json_file(config_path)
        self.embedding = BertModel.from_pretrained(model_path, config=config)
        self.linear = torch.nn.Linear(config.hidden_size, output_size)

        if bert_freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, tokens_type_id, tokens_masks):
        embed = self.embedding(input_ids=tokens,
                               token_type_ids=tokens_type_id,
                               attention_mask=tokens_masks)
        embed = embed.last_hidden_state
        embed = self.linear(embed)
        embed = torch.nn.functional.normalize(embed, p=2, dim=2)
        return embed

    def score(self, question_embeds, document_embeds):
        question_embeds = question_embeds.permute(0, 2, 1)

        logits = torch.matmul(document_embeds, question_embeds)
        logits = torch.max(logits, dim=1).values
        logits = torch.sum(logits, dim=1)

        return logits
