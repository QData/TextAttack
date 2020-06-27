import textattack

from .language_model_constraint import LanguageModelConstraint


class LearningToWriteLanguageModel(LanguageModelConstraint):
    """ A constraint based on the L2W language model.
    
        The RNN-based language model from ``Learning to Write With Cooperative
        Discriminators'' (Holtzman et al, 2018).
        
        https://arxiv.org/pdf/1805.06087.pdf
        https://github.com/windweller/l2w
        
        
        Reused by Jia et al., 2019, as a substitution for the Google 1-billion 
        words language model (in a revised version the attack of Alzantot et 
        al., 2018).
        
        https://worksheets.codalab.org/worksheets/0x79feda5f1998497db75422eca8fcd689
    """

    CACHE_PATH = 'constraints/grammaticality/language-models/learning-to-write'
    def __init__(self, window_size=5, **kwargs):
        # TODO add window size for all LMs
        self.window_size = window_size
        lm_folder_path = utils.download_if_needed(L2WLanguageModel.CACHE_PATH)
        self.query_handler = lmquery.load_model(lm_folder_path, 
            textattack.shared.utils.device)

    def get_log_probs_at_index(self, text_list, word_index):
        """ Gets the probability of the word at index `word_index` according
            to GPT-2. Assumes that all items in `text_list`
            have the same prefix up until `word_index`.
        """
        lm_queries = []
        
        for attacked_text in text_list:
            query_words = attacked_text.text_window_around_index(word_index, self.window_size)
            lm_queries.append(query_words)
        
        log_probs = query_handler.query(queries, batch_size=16)
        
        return log_probs

""" from https://worksheets.codalab.org/worksheets/0x79feda5f1998497db75422eca8fcd689
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  query_handler = lmquery.load_model(device)
  with open(OPTS.neighbor_file) as f:
    neighbors = json.load(f)
  if OPTS.task == 'classification':
    raw_data = text_classification.IMDBDataset.get_raw_data(
        OPTS.imdb_dir, test=(OPTS.split == 'test'))
  elif OPTS.task == 'entailment':
    OPTS.test = (OPTS.split == 'test')
    raw_data = entailment.SNLIDataset.get_raw_data(OPTS)
  else:
    raise NotImplementedError
  if OPTS.split == 'train':
    data = raw_data.train_data
  else:  # dev or test
    data = raw_data.dev_data
  if OPTS.num_examples:
    data = data[:OPTS.num_examples]
  if OPTS.shard is not None:
    print('Restricting to shard %d' % OPTS.shard)
    data = data[OPTS.shard * OPTS.shard_size:(OPTS.shard + 1) * OPTS.shard_size]
  with open(OPTS.out_file, 'w') as f:
    for sent_idx, example in enumerate(tqdm(data)):
      if OPTS.task == 'classification':
        sentence = example[0]
      elif OPTS.task == 'entailment':
        sentence = example[0][1]  # Only look at hypothesis
      print('%d\t%s' % (sent_idx, sentence), file=f)
      words = sentence.split(' ')
      for i, w in enumerate(words):
        if w in neighbors:
          options = [w] + neighbors[w]
          start = max(0, i - OPTS.window_radius)
          end = min(len(words), i + 1 + OPTS.window_radius)
          # Remove OOV words from prefix and suffix
          prefix = [x for x in words[start:i] if x in query_handler.word_to_idx]
          suffix = [x for x in words[i+1:end] if x in query_handler.word_to_idx]
          queries = []
          in_vocab_options = []
          for opt in options:
            if opt in query_handler.word_to_idx:
              queries.append(prefix + [opt] + suffix)
              in_vocab_options.append(opt)
            else:
              print('%d\t%d\t%s\t%s' % (sent_idx, i, opt, float('-inf')), file=f)
          if queries:
            log_probs = query_handler.query(queries, batch_size=16)
            for x, lp in zip(in_vocab_options, log_probs):
              print('%d\t%d\t%s\t%s' % (sent_idx, i, x, lp), file=f)
""""