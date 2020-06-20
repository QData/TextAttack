import torch
import transformers

from textattack.shared import utils

from .language_model_constraint import LanguageModelConstraint


class L2WLanguageModel(LanguageModelConstraint):
    """ A constraint based on the L2W language model.
    
        The RNN-based language model from ``Learning to Write With Cooperative
        Discriminators'' (Holtzman et al, 2018).
        
        https://arxiv.org/pdf/1805.06087.pdf
        https://github.com/windweller/l2w
        
        
        Reused by Jia et al., 2019, as a substitution for the Google 1-billion 
        words language model (in a revised version the attack of Alzantot et 
        al., 2018).
    """

    CACHE_PATH = 'constraints/grammaticality/language-models/learning-to-write'
    def __init__(self, **kwargs):
        model_file = 
        with open(args.lm, 'rb') as model_file:
            model = torch.load(model_file)
        
        model.eval()
        with open(args.dic, 'rb') as dic_file:
            dictionary = pickle.load(dic_file)
        predictor = predictors.RNNPredictor(model, len(dictionary), asm=True)

    def get_log_probs_at_index(self, text_list, word_index):
        """ Gets the probability of the word at index `word_index` according
            to GPT-2. Assumes that all items in `text_list`
            have the same prefix up until `word_index`.
        """
        prefix = text_list[0].text_until_word_index(word_index)

        if not utils.has_letter(prefix):
            # This language model perplexity is not defined with respect to
            # a word without a prefix. If the prefix is null, just return the
            # log-probability 0.0.
            return torch.zeros(len(text_list), dtype=torch.float)

        token_ids = self.tokenizer.encode(prefix)
        tokens_tensor = torch.tensor([token_ids])
        tokens_tensor = tokens_tensor.to(utils.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
        predictions = outputs[0]

        probs = []
        for attacked_text in text_list:
            nxt_word_ids = self.tokenizer.encode(attacked_text.words[word_index])
            next_word_prob = predictions[0, -1, next_word_ids[0]]
            probs.append(next_word_prob)

        return probs
