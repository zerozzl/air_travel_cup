import torch


class Hlatr(torch.nn.Module):
    def __init__(self, input_size, position_size, hidden_size, nhead, nlayers):
        super(Hlatr, self).__init__()

        self.position_embedding = torch.nn.Embedding(position_size, hidden_size)
        self.projection_matrix = torch.nn.Linear(input_size, hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)

        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size), nlayers)
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, tokens, tokens_position):
        embed = self.projection_matrix(tokens)
        embed_pos = self.position_embedding(tokens_position)
        embed = embed + embed_pos
        embed = self.layer_norm(embed)

        logits = self.transformer_encoder(embed)
        logits = self.linear(logits).squeeze(-1)

        return logits
