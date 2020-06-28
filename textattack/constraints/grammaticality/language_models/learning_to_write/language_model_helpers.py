import numpy as np
import os
import torch
import torchfile

from .rnn_model import RNNModel

class QueryHandler():
  def __init__(self, model, word_to_idx, mapto, device):
    self.model = model
    self.word_to_idx = word_to_idx
    self.mapto = mapto
    self.device = device

  def query(self, sentences, swapped_words):
    T = len(sentences[0])
    if any(len(s) != T for s in sentences):
      raise ValueError('Only same length batches are allowed')

    log_probs = []
    for start in range(0, len(sentences)):
      swapped_words_batch = swapped_words[start:min(len(sentences), start + 1)]
      batch = sentences[start:min(len(sentences), start + 1)]
      raw_idx_list = [[] for i in range(T+1)]
      for i, s in enumerate(batch):
        s = [word for word in s if word in self.word_to_idx]
        words = ['<S>'] + s
        word_idxs = [self.word_to_idx[w] for w in words]
        for t in range(T+1):
          if t < len(word_idxs):
            raw_idx_list[t].append(word_idxs[t])
      orig_num_idxs = len(raw_idx_list)
      raw_idx_list = [x for x in raw_idx_list if len(x)]
      num_idxs_dropped = orig_num_idxs - len(raw_idx_list)
      all_raw_idxs = torch.tensor(raw_idx_list, device=self.device,
                                  dtype=torch.long)
      word_idxs = self.mapto[all_raw_idxs]
      hidden = self.model.init_hidden(len(batch))
      source = word_idxs[:-1,:]
      target = word_idxs[1:,:]
      decode, hidden = self.model(source, hidden)
      decode = decode.view(T - num_idxs_dropped, len(batch), -1)
      for i in range(len(batch)):
        if swapped_words_batch[i] not in self.word_to_idx:
          log_probs.append(float('-inf'))
        else:  
          log_probs.append(sum([decode[t, i, target[t, i]].item() for t in range(T-num_idxs_dropped)]))
    return log_probs

def util_reverse(item):
    new_item = np.zeros(len(item))
    for idx, val in enumerate(item):
        new_item[val] = idx
    return new_item

def load_model(lm_folder_path, device):
  word_map = torchfile.load(os.path.join(lm_folder_path, 'word_map.th7'))
  word_map = [w.decode('utf-8') for w in word_map]
  word_to_idx = {w: i for i, w in enumerate(word_map)}
  word_freq = torchfile.load(os.path.join(os.path.join(lm_folder_path, 'word_freq.th7')))
  mapto = torch.from_numpy(util_reverse(np.argsort(-word_freq))).long().to(device)

  model_file = open(os.path.join(lm_folder_path, 'lm-state-dict.pt'), 'rb')
  
  model = RNNModel('GRU', 793471, 256, 2048, 1, [4200, 35000, 180000, 793471], dropout=0.01, proj=True, lm1b=True)
  
  model.load_state_dict(torch.load(model_file))
  model.full = True  # Use real softmax--important!
  model.to(device)
  model.eval()
  model_file.close()
  return QueryHandler(model, word_to_idx, mapto, device)