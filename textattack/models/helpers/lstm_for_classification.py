import torch
from torch import nn as nn

import textattack
from textattack.models.helpers import GloveEmbeddingLayer
from textattack.models.helpers.utils import load_cached_state_dict
from textattack.shared import utils


class LSTMForClassification(nn.Module):
    """A long short-term memory neural network for text classification.

    We use different versions of this network to pretrain models for
    text classification.
    """

    def __init__(
        self,
        hidden_size=150,
        depth=1,
        dropout=0.3,
        num_labels=2,
        max_seq_length=128,
        model_path=None,
        emb_layer_trainable=True,
    ):
        super().__init__()
        if depth <= 1:
            # Fix error where we ask for non-zero dropout with only 1 layer.
            # nn.module.RNN won't add dropout for the last recurrent layer,
            # so if that's all we have, this will display a warning.
            dropout = 0
        self.drop = nn.Dropout(dropout)
        self.emb_layer_trainable = emb_layer_trainable
        self.emb_layer = GloveEmbeddingLayer(emb_layer_trainable=emb_layer_trainable)
        self.word2id = self.emb_layer.word2id
        self.encoder = nn.LSTM(
            input_size=self.emb_layer.n_d,
            hidden_size=hidden_size // 2,
            num_layers=depth,
            dropout=dropout,
            bidirectional=True,
        )
        d_out = hidden_size
        self.out = nn.Linear(d_out, num_labels)
        self.tokenizer = textattack.models.tokenizers.GloveTokenizer(
            word_id_map=self.word2id,
            unk_token_id=self.emb_layer.oovid,
            pad_token_id=self.emb_layer.padid,
            max_length=max_seq_length,
        )

        if model_path is not None:
            self.load_from_disk(model_path)

    def load_from_disk(self, model_path):
        self.load_state_dict(load_cached_state_dict(model_path))
        self.word_embeddings = self.emb_layer.embedding
        self.lookup_table = self.emb_layer.embedding.weight.data
        self.to(utils.device)
        self.eval()

    def forward(self, _input):
        # ensure RNN module weights are part of single contiguous chunk of memory
        self.encoder.flatten_parameters()

        emb = self.emb_layer(_input.t())
        emb = self.drop(emb)

        output, hidden = self.encoder(emb)
        output = torch.max(output, dim=0)[0]

        output = self.drop(output)
        pred = self.out(output)
        return pred
