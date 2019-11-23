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
        self.max_seq_length = max_seq_length
        self.drop = nn.Dropout(dropout)
        self.word_embeddings = GloveEmbeddingLayer()
        self.word2id = self.word_embeddings.word2id

        self.encoder = CNNTextLayer(
            self.word_embeddings.n_d,
            widths = [3,4,5],
            filters=hidden_size
        )
        d_out = 3*hidden_size
        self.out = nn.Linear(d_out, nclasses)
    
    def load_from_disk(self, model_path):
        state_dict = torch.load(model_path, map_location=utils.get_device())
        self.load_state_dict(state_dict)
        self.to(utils.get_device())
        self.eval()

    def forward(self, _input):
        emb = self.word_embeddings(_input)
        emb = self.drop(emb)

        output = self.encoder(emb)

        output = self.drop(output)
        pred = self.out(output)
        return nn.functional.softmax(pred, dim=-1)
    
    def convert_text_to_ids(self, input_text):
        input_tokens = utils.default_tokenize(input_text)
        input_tokens = input_tokens[:self.max_seq_length]
        output_ids = []
        for word in input_tokens:
            if word in self.word2id:
                output_ids.append(self.word2id[word])
            else:
                output_ids.append(self.word_embeddings.oovid)
        zeros_to_add = self.max_seq_length - len(output_ids)
        output_ids += [self.word_embeddings.padid] * zeros_to_add
        return output_ids

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