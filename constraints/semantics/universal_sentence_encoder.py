from .models import InferSent
from .utils import cache_path

import numpy as np
import torch

class UniversalSentenceEncoder(constraint):
    """ Constraint using cosine similarity between Universal Sentence Encodings
        of x and x_adv.
        
        Uses InferSent sentence embeddings. """
    
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.model = self.get_infersent_model()
    
    def get_infersent_model(self):
        infersent_version = 2
        MODEL_PATH = cache_path(f'infersent-encoder/infersent{infersent_version}.pkl')
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': infersent_version}
        infersent = InferSent(params_model)
        infersent.load_state_dict(torch.load(MODEL_PATH))
        W2V_PATH = 'word_embeddings/fastText/crawl-300d-2M.vec'
        infersent.set_w2v_path(W2V_PATH)
        infersent.build_vocab_k_words(K=100000)
        return infersent
    
    def score(self, x, x_adv):
        """ Returns the cosine similarity between embeddings of text x and 
            x_adv. 
            
            @TODO should this support multiple sentences for x_adv?
        """
        original_embedding, perturbed_embedding = self.model.encode([x, x_adv], tokenize = True)[0]
        
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(original_embedding, perturbed_embedding)
        
        
    def OLD_DLEETE_AFTER_RECREATING_IN_SEARCH_score(self, text, index_to_replace, candidates, tokenized=False,
            recover_adv=default_recover_adv):
        """ Returns cosine similarity between the USE encodings of x and x_adv.
        """
        
        raw_sentences = np.array(text, dtype=object)
        raw_sentences = np.tile(raw_sentences, (len(candidates), 1))
        raw_sentences[list(range(len(candidates))), np.tile(index_to_replace, len(candidates))] = np.array(candidates)
        raw_sentences = list(map(lambda s: recover_adv(s), raw_sentences))
        
        original_embedding = self.model.encode([recover_adv(text)], tokenize = True)[0]
        altered_embeddings = self.model.encode(raw_sentences, tokenize = True)
        
        cos = torch.nn.CosineSimilarity(dim=0)
        def cos_similarity(embedding):
            return cos(torch.from_numpy(original_embedding), torch.from_numpy(embedding))
        
        return list(np.apply_along_axis(cos_similarity, 1, altered_embeddings))
    
    def __call__(self, x, x_adv):
        return self.score(x, x_adv) >= self.threshold