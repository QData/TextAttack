import os
import textattack
import torch
import torch.nn as nn
import torch.nn.functional as F

import textattack.utils as utils
from textattack.models.helpers import GloveEmbeddingLayer

class WordCNNForClassification(nn.Module):
    """ A convolutional neural network for text classification. 
    
        We use different versions of this network to pretrain models for text 
        classification.
    """
    def __init__(self, hidden_size=150, dropout=0.3, nclasses=2, max_seq_length=128):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.emb_layer = GloveEmbeddingLayer()
        self.word2id = self.emb_layer.word2id
        self.encoder = CNNTextLayer(
            self.emb_layer.n_d,
            widths = [3,4,5],
            filters=hidden_size
        )
        d_out = 3*hidden_size
        self.out = nn.Linear(d_out, nclasses)
        self.tokenizer = textattack.tokenizers.SpacyTokenizer(self.word2id,
            self.emb_layer.oovid, self.emb_layer.padid, max_seq_length)
    
    def load_from_disk(self, model_folder_path):
        model_folder_path = utils.download_if_needed(model_folder_path)
        model_path = os.path.join(model_folder_path, 'model.bin')
        state_dict = torch.load(model_path, map_location=utils.get_device())
        self.load_state_dict(state_dict)
        self.to(utils.get_device())
        self.eval()

    def forward(self, _input):
        emb = self.emb_layer(_input)
        emb = self.drop(emb)

        output = self.encoder(emb)

        output = self.drop(output)
        pred = self.out(output)
        return nn.functional.softmax(pred, dim=-1)

class CNNTextLayer(nn.Module):
    def __init__(self, n_in, widths=[3,4,5], filters=100):
        super().__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x):
        x = x.unsqueeze(1) # (batch, Ci, len, d)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(batch, Co, len), ...]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]
        x = torch.cat(x, 1)
        return x