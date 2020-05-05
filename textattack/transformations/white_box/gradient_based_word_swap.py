import torch

from textattack.shared import utils
from textattack.transformations import Transformation

class GradientBasedWordSwap(Transformation):
    """ Uses the model's gradient to suggest replacements for a given word.
        
        Based off of HotFlip: White-Box Adversarial Examples for Text 
            Classification (Ebrahimi et al., 2018).
        
            https://arxiv.org/pdf/1712.06751.pdf
        
        Arguments:
            model (nn.Module): The model to attack. Model must have a 
                `word_embeddings` matrix and `convert_id_to_word` function.
            top_n (int): the number of top words to return at each index
            replace_stopwords (bool): whether or not to replace stopwords
    """
    def __init__(self, model, top_n=1, replace_stopwords=False):
        # @TODO validate model is our word LSTM here
        if not hasattr(model, 'word_embeddings'):
            raise ValueError('Model needs word embedding matrix for gradient-based word swap')
        if not hasattr(model, 'lookup_table'):
            raise ValueError('Model needs lookup table for gradient-based word swap')
        if not hasattr(model, 'zero_grad'):
            raise ValueError('Model needs `zero_grad()` for gradient-based word swap')
        if not hasattr(model.tokenizer, 'convert_id_to_word'):
            raise ValueError('Tokenizer needs `convert_id_to_word()` for gradient-based word swap')
        if not hasattr(model.tokenizer, 'pad_id'):
            raise ValueError('Tokenizer needs `pad_id` for gradient-based word swap')
        if not hasattr(model.tokenizer, 'oov_id'):
            raise ValueError('Tokenizer needs `oov_id` for gradient-based word swap')
        self.loss = torch.nn.CrossEntropyLoss()
        self.model = model
        self.pad_id = self.model.tokenizer.pad_id
        self.oov_id = self.model.tokenizer.oov_id
        self.top_n = top_n
        # @TODO optionally take other loss functions as a param.
        if replace_stopwords:
            self.stopwords = set()
        else:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))

    def _replace_word_at_index(self, text, word_index):
        """ Returns returns a list containing all possible words to replace
            `word` with, based off of the model's gradient.
            
            Arguments:
                text (TokenizedText): The full text input to perturb
                word_index (int): index of the word to replace
        """
        self.model.train()
       
        lookup_table = self.model.lookup_table.to(utils.get_device())
        lookup_table_transpose = lookup_table.transpose(0,1)
        
        # set backward hook on the word embeddings for input x
        emb_hook = Hook(self.model.word_embeddings, backward=True)
    
        self.model.zero_grad()
        predictions = self._call_model(text)
        original_label = predictions.argmax() # @TODO is this right? Do we need to pass in `original_label`?
        y_true = torch.Tensor([original_label]).long().to(utils.get_device())
        loss = self.loss(predictions, y_true)
        loss.backward()
    
        # grad w.r.t to word embeddings
        emb_grad = emb_hook.output[0].to(utils.get_device()).squeeze()
    
        # grad differences between all flips and original word (eq. 1 from paper)
        vocab_size = lookup_table.size(0)
        
        # Get the grad w.r.t the one-hot index of the word.
        b_grads = emb_grad[word_index].view(1,-1).mm(lookup_table_transpose).squeeze()
        a_grad = b_grads[text.ids[0][word_index]]
        diffs = b_grads-a_grad
        
        # Don't change to the pad token.
        diffs[self.model.tokenizer.pad_id] = 0
        
        word_idxs_sorted_by_grad = (-diffs).argsort()[:self.top_n]
        
        candidate_words = []
        for word_id in word_idxs_sorted_by_grad:
            candidate_words.append(self.model.tokenizer.convert_id_to_word(word_id.item()))
            
        self.model.eval()
        return candidate_words
    
    def _call_model(self, text):
        """ A helper function to query `self.model` with TokenizedText `text`.
        """
        ids = torch.tensor(text.ids[0])
        ids = ids.to(next(self.model.parameters()).device)
        ids = ids.unsqueeze(0)
        return self.model(ids)

    def __call__(self, tokenized_text, indices_to_replace=None):
        """
        Returns a list of all possible transformations for `text`.
            
        If indices_to_replace is set, only replaces words at those indices.
        
        """
        words = tokenized_text.words
        if not indices_to_replace:
            indices_to_replace = list(range(len(words)))
        
        transformations = []
        word_swaps = []
        for i in indices_to_replace:
            word_to_replace = words[i]
            # Don't replace stopwords.
            if word_to_replace.lower() in self.stopwords:
                continue
            replacement_words = self._replace_word_at_index(tokenized_text, i)
            new_tokenized_texts = []
            for r in replacement_words:
                # Don't replace with numbers, punctuation, or other non-letter characters.
                # if not is_word(r):
                    # continue
                new_tokenized_texts.append(tokenized_text.replace_word_at_index(i, r))
            transformations.extend(new_tokenized_texts)
        return transformations

class Hook:
    def __init__(self, module, backward=False):
        if backward:
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_hook(self.hook_fn)
            
    def hook_fn(self, module, input, output):
        self.input = [x.to(utils.get_device()) for x in input]
        self.output = [x.to(utils.get_device()) for x in output]
        
    def close(self):
        self.hook.remove()