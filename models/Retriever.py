import torch
from transformers import BertConfig, BertModel


class DprBert(torch.nn.Module):
    def __init__(self, config_path, model_path, bert_freeze):
        super(DprBert, self).__init__()

        config = BertConfig.from_json_file(config_path)
        self.question_encoder = BertModel.from_pretrained(model_path, config=config)
        self.context_encoder = BertModel.from_pretrained(model_path, config=config)

        if bert_freeze:
            for param in self.question_encoder.parameters():
                param.requires_grad = False
            for param in self.context_encoder.parameters():
                param.requires_grad = False

    def forward(self, questions, questions_seg, contexts, contexts_seg):
        questions_mask = (questions > 0).int()
        contexts_mask = (contexts > 0).int()

        questions_rep = self.question_encoder(input_ids=questions,
                                              token_type_ids=questions_seg,
                                              attention_mask=questions_mask)
        questions_rep = questions_rep.last_hidden_state
        questions_rep = questions_rep[:, 0, :]

        contexts_rep = self.context_encoder(input_ids=contexts,
                                            token_type_ids=contexts_seg,
                                            attention_mask=contexts_mask)
        contexts_rep = contexts_rep.last_hidden_state
        contexts_rep = contexts_rep[:, 0, :]

        scores = torch.matmul(questions_rep, torch.transpose(contexts_rep, 0, 1))
        return scores

    def get_question_embedding(self, questions, questions_seg):
        questions_mask = (questions > 0).int()
        questions_rep = self.question_encoder(input_ids=questions,
                                              token_type_ids=questions_seg,
                                              attention_mask=questions_mask)
        questions_rep = questions_rep.last_hidden_state
        questions_rep = questions_rep[:, 0, :]
        return questions_rep

    def get_context_embedding(self, contexts, contexts_seg):
        contexts_mask = (contexts > 0).int()
        contexts_rep = self.context_encoder(input_ids=contexts,
                                            token_type_ids=contexts_seg,
                                            attention_mask=contexts_mask)
        contexts_rep = contexts_rep.last_hidden_state
        contexts_rep = contexts_rep[:, 0, :]
        return contexts_rep


class MvrBert(torch.nn.Module):
    def __init__(self, config_path, model_path, view_num, bert_freeze):
        super(MvrBert, self).__init__()

        config = BertConfig.from_json_file(config_path)
        self.question_encoder = BertModel.from_pretrained(model_path, config=config)
        self.context_encoder = BertModel.from_pretrained(model_path, config=config)
        self.view_num = view_num

        if bert_freeze:
            for param in self.question_encoder.parameters():
                param.requires_grad = False
            for param in self.context_encoder.parameters():
                param.requires_grad = False

    def forward(self, question_tokens, question_tokens_type_id, question_position_ids,
                document_tokens, document_tokens_type_id, document_position_ids):
        questions_mask = (question_tokens > 0).int()
        contexts_mask = (document_tokens > 0).int()

        questions_rep = self.question_encoder(input_ids=question_tokens,
                                              token_type_ids=question_tokens_type_id,
                                              attention_mask=questions_mask,
                                              position_ids=question_position_ids)
        questions_rep = questions_rep.last_hidden_state
        questions_rep = questions_rep[:, 0, :]

        contexts_rep = self.context_encoder(input_ids=document_tokens,
                                            token_type_ids=document_tokens_type_id,
                                            attention_mask=contexts_mask,
                                            position_ids=document_position_ids)
        contexts_rep = contexts_rep.last_hidden_state
        contexts_rep = contexts_rep[:, 0:self.view_num, :]

        scores = torch.matmul(questions_rep, torch.transpose(contexts_rep, 2, 1))
        scores = torch.transpose(scores, 1, 0)

        max_scores, _ = torch.max(scores, dim=2)
        return scores, max_scores

    def get_question_embedding(self, question_tokens, question_tokens_type_id, question_position_ids):
        questions_mask = (question_tokens > 0).int()
        questions_rep = self.question_encoder(input_ids=question_tokens,
                                              token_type_ids=question_tokens_type_id,
                                              attention_mask=questions_mask,
                                              position_ids=question_position_ids)
        questions_rep = questions_rep.last_hidden_state
        questions_rep = questions_rep[:, 0, :]
        return questions_rep

    def get_context_embedding(self, document_tokens, document_tokens_type_id, document_position_ids):
        contexts_mask = (document_tokens > 0).int()
        contexts_rep = self.context_encoder(input_ids=document_tokens,
                                            token_type_ids=document_tokens_type_id,
                                            attention_mask=contexts_mask,
                                            position_ids=document_position_ids)
        contexts_rep = contexts_rep.last_hidden_state
        contexts_rep = contexts_rep[:, 0:self.view_num, :]
        return contexts_rep
