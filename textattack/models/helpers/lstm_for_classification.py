import textattack
import torch
import torch.nn as nn

from textattack.shared import utils

from textattack.models.helpers import GloveEmbeddingLayer
from textattack.models.helpers.utils import load_cached_state_dict

class LSTMForClassification(nn.Module):
    """ A long short-term memory neural network for text classification. 
    
        We use different versions of this network to pretrain models for text 
        classification.
    """
    def __init__(self, hidden_size=150, depth=1, dropout=0.3, nclasses=2,
        max_seq_length=128):
        super().__init__()
        if depth <= 1:
            # Fix error where we ask for non-zero dropout with only 1 layer.
            # nn.module.RNN won't add dropout for the last recurrent layer,
            # so if that's all we have, this will display a warning.
            dropout = 0
        self.drop = nn.Dropout(dropout)
        self.emb_layer = GloveEmbeddingLayer()
        self.word2id = self.emb_layer.word2id
        self.encoder = nn.LSTM(
            input_size=self.emb_layer.n_d,
            hidden_size=hidden_size//2,
            num_layers=depth,
            dropout=dropout,
            bidirectional=True
        )
        d_out = hidden_size
        self.out = nn.Linear(d_out, nclasses)
        self.tokenizer = textattack.tokenizers.SpacyTokenizer(self.word2id,
            self.emb_layer.oovid, self.emb_layer.padid, max_seq_length)
    
    def load_from_disk(self, model_folder_path):
        self.load_state_dict(load_cached_state_dict(model_folder_path))
        self.word_embeddings = self.emb_layer.embedding
        self.lookup_table = self.emb_layer.embedding.weight.data
        self.to(utils.get_device())
        self.eval()

    def forward(self, _input):
        emb = self.emb_layer(_input.t())
        emb = self.drop(emb)
        
        output, hidden = self.encoder(emb)
        output = torch.max(output, dim=0)[0]

        output = self.drop(output)
        pred = self.out(output)
        return nn.functional.softmax(pred, dim=-1)
        