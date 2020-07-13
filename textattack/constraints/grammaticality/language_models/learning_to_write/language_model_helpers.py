import os

import numpy as np
import torch
import torchfile

from .rnn_model import RNNModel


class QueryHandler:
    def __init__(self, model, word_to_idx, mapto, device):
        self.model = model
        self.word_to_idx = word_to_idx
        self.mapto = mapto
        self.device = device

    def query(self, sentences, swapped_words, batch_size=32):
        """Since we don't filter prefixes for OOV ahead of time, it's possible
        that some of them will have different lengths. When this is the case,
        we can't do RNN prediction in batch.

        This method _tries_ to do prediction in batch, and, when it
        fails, just does prediction sequentially and concatenates all of
        the results.
        """
        try:
            return self.try_query(sentences, swapped_words, batch_size=batch_size)
        except Exception:
            probs = []
            for s, w in zip(sentences, swapped_words):
                try:
                    probs.append(self.try_query([s], [w], batch_size=1)[0])
                except RuntimeError:
                    print(
                        "WARNING:  got runtime error trying languag emodel on language model w s/w",
                        s,
                        w,
                    )
                    probs.append(float("-inf"))
            return probs

    def try_query(self, sentences, swapped_words, batch_size=32):
        # TODO use caching
        sentence_length = len(sentences[0])
        if any(len(s) != sentence_length for s in sentences):
            raise ValueError("Only same length batches are allowed")

        log_probs = []
        for start in range(0, len(sentences), batch_size):
            swapped_words_batch = swapped_words[
                start : min(len(sentences), start + batch_size)
            ]
            batch = sentences[start : min(len(sentences), start + batch_size)]
            raw_idx_list = [[] for i in range(sentence_length + 1)]
            for i, s in enumerate(batch):
                s = [word for word in s if word in self.word_to_idx]
                words = ["<S>"] + s
                word_idxs = [self.word_to_idx[w] for w in words]
                for t in range(sentence_length + 1):
                    if t < len(word_idxs):
                        raw_idx_list[t].append(word_idxs[t])
            orig_num_idxs = len(raw_idx_list)
            raw_idx_list = [x for x in raw_idx_list if len(x)]
            num_idxs_dropped = orig_num_idxs - len(raw_idx_list)
            all_raw_idxs = torch.tensor(
                raw_idx_list, device=self.device, dtype=torch.long
            )
            word_idxs = self.mapto[all_raw_idxs]
            hidden = self.model.init_hidden(len(batch))
            source = word_idxs[:-1, :]
            target = word_idxs[1:, :]
            if (not len(source)) or not len(hidden):
                return [float("-inf")] * len(batch)
            decode, hidden = self.model(source, hidden)
            decode = decode.view(sentence_length - num_idxs_dropped, len(batch), -1)
            for i in range(len(batch)):
                if swapped_words_batch[i] not in self.word_to_idx:
                    log_probs.append(float("-inf"))
                else:
                    log_probs.append(
                        sum(
                            [
                                decode[t, i, target[t, i]].item()
                                for t in range(sentence_length - num_idxs_dropped)
                            ]
                        )
                    )
        return log_probs

    @staticmethod
    def load_model(lm_folder_path, device):
        word_map = torchfile.load(os.path.join(lm_folder_path, "word_map.th7"))
        word_map = [w.decode("utf-8") for w in word_map]
        word_to_idx = {w: i for i, w in enumerate(word_map)}
        word_freq = torchfile.load(
            os.path.join(os.path.join(lm_folder_path, "word_freq.th7"))
        )
        mapto = torch.from_numpy(util_reverse(np.argsort(-word_freq))).long().to(device)

        model_file = open(os.path.join(lm_folder_path, "lm-state-dict.pt"), "rb")

        model = RNNModel(
            "GRU",
            793471,
            256,
            2048,
            1,
            [4200, 35000, 180000, 793471],
            dropout=0.01,
            proj=True,
            lm1b=True,
        )

        model.load_state_dict(torch.load(model_file, map_location=device))
        model.full = True  # Use real softmax--important!
        model.to(device)
        model.eval()
        model_file.close()
        return QueryHandler(model, word_to_idx, mapto, device)


def util_reverse(item):
    new_item = np.zeros(len(item))
    for idx, val in enumerate(item):
        new_item[val] = idx
    return new_item
