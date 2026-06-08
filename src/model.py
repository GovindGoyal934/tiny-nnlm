import torch
import torch.nn as nn
import config


class NNLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=config.EMB_DIM,
        context_window=config.CONTEXT_WINDOW,
        hidden_dim=config.HIDDEN_DIM,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_window = context_window
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        emb_concat_dim = context_window * emb_dim

        self.hidden = nn.Linear(emb_concat_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=getattr(config, "DROPOUT", 0.0))

        # Direct connections (skip-connections) from embeddings to output
        self.direct = nn.Linear(emb_concat_dim, vocab_size)
        # Hidden-to-output
        self.output = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -1.0 / self.emb_dim, 1.0 / self.emb_dim)
        nn.init.uniform_(self.hidden.weight, -0.1, 0.1)
        nn.init.zeros_(self.hidden.bias)
        nn.init.uniform_(self.direct.weight, -0.1, 0.1)
        nn.init.zeros_(self.direct.bias)
        nn.init.uniform_(self.output.weight, -0.1, 0.1)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(emb.size(0), -1)
        emb = self.dropout(emb)
        h = self.activation(self.hidden(emb))
        h = self.dropout(h)
        out = self.direct(emb) + self.output(h)
        return out
