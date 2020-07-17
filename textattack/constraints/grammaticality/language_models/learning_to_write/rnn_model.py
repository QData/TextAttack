from torch import nn as nn
from torch.autograd import Variable

from .adaptive_softmax import AdaptiveSoftmax


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.

    Based on official pytorch examples
    """

    def __init__(
        self,
        rnn_type,
        ntoken,
        ninp,
        nhid,
        nlayers,
        cutoffs,
        proj=False,
        dropout=0.5,
        tie_weights=False,
        lm1b=False,
    ):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.lm1b = lm1b

        if rnn_type == "GRU":
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
            )

        self.proj = proj

        if ninp != nhid and proj:
            self.proj_layer = nn.Linear(nhid, ninp)

        # if tie_weights:
        #     if nhid != ninp and not proj:
        #         raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder = nn.Linear(ninp, ntoken)
        #     self.decoder.weight = self.encoder.weight
        # else:
        #     if nhid != ninp and not proj:
        #         if not lm1b:
        #             self.decoder = nn.Linear(nhid, ntoken)
        #         else:
        #             self.decoder = adapt_loss
        #     else:
        #         self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        if proj:
            self.softmax = AdaptiveSoftmax(ninp, cutoffs)
        else:
            self.softmax = AdaptiveSoftmax(nhid, cutoffs)

        self.full = False

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)

        if "proj" in vars(self):
            if self.proj:
                output = self.proj_layer(output)

        output = output.view(output.size(0) * output.size(1), output.size(2))

        if self.full:
            decode = self.softmax.log_prob(output)
        else:
            decode = self.softmax(output)

        return decode, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
