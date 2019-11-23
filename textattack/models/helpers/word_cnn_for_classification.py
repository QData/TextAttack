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
        self.emb_layer = GloveEmbeddingLayer()

        self.encoder = CNNTextLayer(
            self.emb_layer.n_d,
            widths = [3,4,5],
            filters=hidden_size
        )
        d_out = 3*hidden_size
        self.out = nn.Linear(d_out, nclasses)
    
    def load_from_disk(self, model_path):
        state_dict = torch.load(model_path, map_location=utils.get_device())
        self.load_state_dict(state_dict)
        self.word_embeddings = self.emb_layer
        self.lookup_table = self.word_embeddings.embedding
        self.to(utils.get_device())
        self.eval()

    def forward(self, _input):
        emb = self.word_embeddings(_input)
        emb = self.drop(emb)

        output = self.encoder(emb)

        output = self.drop(output)
        pred = self.out(output)
        return nn.functional.softmax(pred, dim=-1)
    
    def convert_text_to_tokens(self, input_text):
        tokens = utils.default_tokenize(input_text)
        tokens = tokens[:self.max_seq_length]
        pad_tokens_to_add = self.max_seq_length - len(tokens)
        tokens += [self.word_embeddings.padid] * pad_tokens_to_add
        return tokens
        
    def convert_tokens_to_ids(self, tokens):
        output_ids = []
        for token in tokens:
            if token in self.emb_layer.word2id:
                output_ids.append(self.emb_layer.word2id[token])
            else:
                output_ids.append(self.emb_layer.oovid)
        zeros_to_add = self.max_seq_length - len(output_ids)
        output_ids += [0] * zeros_to_add
        return output_ids
    
    def convert_id_to_word(self, text):
        return self.word_embeddings.id2word[text]
        

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