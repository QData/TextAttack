import torch
import torch.nn as nn

import textattack.utils as utils
from textattack.models.helpers import GloveEmbeddingLayer


class LSTMForClassification(nn.Module):
    """ A long short-term memory neural network for text classification. 
    
        We use different versions of this network to pretrain models for text 
        classification.
    """
    def __init__(self, hidden_size=150, depth=1, dropout=0.3, nclasses=2,
        max_seq_length=128):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.drop = nn.Dropout(dropout)
        self.emb_layer = GloveEmbeddingLayer()
        self.word2id = self.emb_layer.word2id
        self.encoder = nn.LSTM(
            self.emb_layer.n_d,
            hidden_size//2,
            depth,
            dropout = dropout,
            bidirectional=True
        )
        d_out = hidden_size
        self.out = nn.Linear(d_out, nclasses)
    
    def load_from_disk(self, model_path):
        state_dict = torch.load(model_path, map_location=utils.get_device())
        self.load_state_dict(state_dict)
        self.to(utils.get_device())
        self.eval()

    def forward(self, _input):
        emb = self.emb_layer(_input.t())
        emb = self.drop(emb)
        
        output, hidden = self.encoder(emb)
        output = torch.max(output, dim=0)[0]

        output = self.drop(output)
        return self.out(output)
    
    def convert_text_to_ids(self, input_text):
        input_tokens = utils.default_tokenize(input_text)
        input_tokens = input_tokens[:self.max_seq_length]
        output_ids = []
        for word in input_tokens:
            if word in self.word2id:
                output_ids.append(self.word2id[word])
            else:
                output_ids.append(self.emb_layer.oovid)
        zeros_to_add = self.max_seq_length - len(output_ids)
        output_ids += [self.emb_layer.padid] * zeros_to_add
        return output_ids

    def text_pred(self, text, batch_size=32):
        batches_x = dataloader.create_batches_x(
            text,
            batch_size,
            self.word2id
        )
        outs = []
        with torch.no_grad():
            for x in batches_x:
                x = Variable(x)
                emb = self.emb_layer(x)
                output, hidden = self.encoder(emb)
                output = torch.max(output, dim=0)[0]
                outs.append(F.softmax(self.out(output), dim=-1))

        return torch.cat(outs, dim=0)