import torch
import torchfile

class QueryHandler():
  def __init__(self, model, word_to_idx, mapto, device):
    self.model = model
    self.word_to_idx = word_to_idx
    self.mapto = mapto
    self.device = device

  def query(self, sentences, batch_size=1):
    T = len(sentences[0])
    if any(len(s) != T for s in sentences):
      raise ValueError('Only same length batches are allowed')

    log_probs = []
    for start in range(0, len(sentences), batch_size):
      batch = sentences[start:min(len(sentences), start + batch_size)]
      raw_idx_list = [[] for i in range(T+1)]
      for i, s in enumerate(batch):
        words = ['<S>'] + s
        word_idxs = [self.word_to_idx[w] for w in words]
        for t in range(T+1):
          raw_idx_list[t].append(word_idxs[t])
      all_raw_idxs = torch.tensor(raw_idx_list, device=self.device,
                                  dtype=torch.long)
      word_idxs = self.mapto[all_raw_idxs]
      hidden = self.model.init_hidden(len(batch))
      source = word_idxs[:-1,:]
      target = word_idxs[1:,:]
      decode, hidden = self.model(source, hidden)
      decode = decode.view(T, len(batch), -1)
      for i in range(len(batch)):
        log_probs.append(sum([decode[t, i, target[t, i]].item() for t in range(T)]))
    return log_probs


def load_model(lm_folder_path, device):
  word_map = torchfile.load(os.path.join(lm_folder_path, 'word_map.th7'))
  word_map = [w.decode('utf-8') for w in word_map]
  word_to_idx = {w: i for i, w in enumerate(word_map)}
  word_freq = torchfile.load(os.path.join(os.path.join(lm_folder_path, 'word_freq.th7')))
  mapto = torch.from_numpy(util.reverse(np.argsort(-word_freq))).long().to(device)

  with open(os.path.join(lm_folder_path, 'lm.pt'), 'rb') as model_file:
    model = torch.load(model_file)
  model.full = True  # Use real softmax--important!
  model.to(device)
  model.eval()
  return QueryHandler(model, word_to_idx, mapto, device)