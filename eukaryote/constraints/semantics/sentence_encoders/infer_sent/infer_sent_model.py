"""
Infer sent model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This file contains the definition of encoders used in
https://arxiv.org/pdf/1705.02364.pdf.

"""


# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time

import numpy as np
import torch
from torch import nn as nn

import eukaryote


class InferSentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bsize = config["bsize"]
        self.word_emb_dim = config["word_emb_dim"]
        self.enc_lstm_dim = config["enc_lstm_dim"]
        self.pool_type = config["pool_type"]
        self.dpout_model = config["dpout_model"]
        self.version = 1 if "version" not in config else config["version"]

        self.enc_lstm = nn.LSTM(
            self.word_emb_dim,
            self.enc_lstm_dim,
            1,
            bidirectional=True,
            dropout=self.dpout_model,
        )

        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = "<s>"
            self.eos = "</s>"
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = "<p>"
            self.eos = "</p>"
            self.max_pad = False
            self.moses_tok = True

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: (seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = (
            torch.from_numpy(idx_sort).to(eukaryote.shared.utils.device)
            if self.is_cuda()
            else torch.from_numpy(idx_sort)
        )
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = (
            torch.from_numpy(idx_unsort).to(eukaryote.shared.utils.device)
            if self.is_cuda()
            else torch.from_numpy(idx_unsort)
        )
        sent_output = sent_output.index_select(1, idx_unsort)

        # Pooling
        if self.pool_type == "mean":
            sent_len = (
                torch.FloatTensor(sent_len.copy())
                .unsqueeze(1)
                .to(eukaryote.shared.utils.device)
            )
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ""
        word_dict[self.bos] = ""
        word_dict[self.eos] = ""
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, "w2v_path"), "w2v path not set"
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding="utf-8") as f:
            for line in f:
                word, vec = line.split(" ", 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=" ")
        print("Found %s(/%s) words with w2v vectors" % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, "w2v_path"), "w2v path not set"
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding="utf-8") as f:
            for line in f:
                word, vec = line.split(" ", 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=" ")
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=" ")

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, "w2v_path"), "w2v path not set"
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        # print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, "w2v_path"), "w2v path not set"
        self.word_vec = self.get_w2v_k(K)
        # print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, "w2v_path"), "warning : w2v path not set"
        assert hasattr(self, "word_vec"), "build_vocab before updating it"
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print(
            "New vocab size : %s (added %s words)"
            % (len(self.word_vec), len(new_word_vec))
        )

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize

        if self.moses_tok:
            s = " ".join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [
            [self.bos] + s.split() + [self.eos]
            if not tokenize
            else [self.bos] + self.tokenize(s) + [self.eos]
            for s in sentences
        ]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings

                warnings.warn(
                    'No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..'
                    % (sentences[i], i)
                )
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print(
                "Nb words kept : %s/%s (%.1f%s)" % (n_wk, n_w, 100.0 * n_wk / n_w, "%")
            )

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
            sentences, bsize, tokenize, verbose
        )

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = self.get_batch(sentences[stidx : stidx + bsize])
            if self.is_cuda():
                batch = batch.to(eukaryote.shared.utils.device)
            with torch.no_grad():
                batch = (
                    self.forward((batch, lengths[stidx : stidx + bsize]))
                    .data.cpu()
                    .numpy()
                )
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print(
                "Speed : %.1f sentences/s (%s mode, bsize=%s)"
                % (
                    len(embeddings) / (time.time() - tic),
                    "gpu" if self.is_cuda() else "cpu",
                    bsize,
                )
            )
        return embeddings
