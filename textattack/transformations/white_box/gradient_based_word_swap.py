import numpy as np

from textattack.shared import utils
from textattack.transformations.word_swap import WordSwap

class GradientBasedWordSwap(WordSwap):
    """ Uses the model's gradient to suggest replacements for a given word.
        
        Based off of HotFlip: White-Box Adversarial Examples for Text 
            Classification (Ebrahimi et al., 2018).
        
            https://arxiv.org/pdf/1712.06751.pdf
        
        Arguments:
            model (nn.Module): The model to attack. Model must have a 
                `word_embeddings` matrix and `convert_id_to_word` function.
            max_swaps (int): The maximum number of words to swap before giving
                up. Called `r` in the HotFlip paper.
    """
    def __init__(self, model, **kwargs):
        # @TODO validate model is our word LSTM here
        if not hasattr(model, 'word_embeddings'):
            raise ValueError('Model needs word embedding matrix for gradient-based word swap')
        if not hasattr(model, 'lookup_table'):
            raise ValueError('Model needs lookup table for gradient-based word swap')
        if not hasattr(model, 'zero_grad'):
            raise ValueError('Model needs `zero_grad()` for gradient-based word swap')
        if not hasattr(model.tokenizer, 'convert_id_to_word'):
            raise ValueError('Tokenizer needs `convert_id_to_word()` for gradient-based word swap')
        if not hasattr(model.tokenizer, 'pad_token_id'):
            raise ValueError('Tokenizer needs `pad_token_id` for gradient-based word swap')
        super().__init__(model)
        self.max_swaps = max_swaps
        self.loss = torch.nn.CrossEntropyLoss()
        self.projection = WordGradientProjection()
        self.pad_id = self.model.tokenizer.pad_token_id
        self.oov_id = self.model.tokenizer.encode('<oov>')[0]
        print('oov', self.model.tokenizer.encode('<oov>'), 'or', 
            self.model.tokenizer.encode('<oov>')[0])
        # @TODO optionally take other loss functions as a param.
        # @TODO move projection into its own class
        # @TODO call self.get_transformations() so constraints are applied
        # @TODO add word-embedding-distance constraint for hotflip
        # @TODO make recipe file for hotflip
        # self.projection = WordGradientProjection()
        super().__init__(**kwargs)

    def _get_replacement_words(self, word):
        """ Returns returns a list containing all possible words to replace
            `word` with, based off of the model's gradient.
        """
       # set backward hook on the word embeddings for input x
        emb_hook = Hook(self.model.word_embeddings, backward=True)
    
        self.model.zero_grad()
        predictions = self._call_model(text)
        loss = self.loss(predictions, y_true)
        loss.backward()
    
        # grad w.r.t to word embeddings
        emb_grad = emb_hook.output[0].to(get_device()).squeeze()
    
        # grad differences between all flips and original word (eq. 1 from paper)
        vocab_size = lookup_table.size(0)
        diffs = torch.zeros(len(swappable_token_idxs), vocab_size)
        for j, word_idx in enumerate(swappable_token_idxs):
            # Get the grad w.r.t the one-hot index of the word.
            b_grads = emb_grad[word_idx].view(1,-1).mm(lookup_table_transpose).squeeze()
            a_grad = b_grads[text.ids[word_idx]]
            diffs[j] = b_grads-a_grad
        
        # Don't change to the pad token.
        diffs[:, self.model.tokenizer.pad_token_id] = 0
        
        import pdb; pdb.set_trace()
        all_words_max_vals, all_words_max_flips = diffs.max(1)
        max_diff, max_word_idx = all_words_max_vals.max(0)
        max_word_flip = all_words_max_flips[max_word_idx]
        
        max_word_idx_in_text = swappable_token_idxs[max_word_idx]
        
        # Only swap the word at this index once.
        swappable_token_idxs.remove(max_word_idx_in_text)
    
        # max_word_idx_in_text = max_word_idx_in_text.item() # the index of the word we should flip from x_tensor
        max_word_flip = max_word_flip.item() # the word to flip max_word_idx to 
        
        new_token = self.model.tokenizer.convert_id_to_word(max_word_flip)
        print('max_word_flip:',max_word_flip,'new_token:',new_token)
        print('replacing idx:', max_word_idx_in_text,'/', text.tokens[max_word_idx_in_text],'with', new_token)
        text = text.replace_token_at_index(max_word_idx_in_text, new_token)
        
        scores = self._call_model(text).squeeze()
        print('\t scores:', scores)
        new_text_label = scores.argmax().item()
        
        if new_text_label != original_label:
            print('hotflip succeeded after', swaps, 'swaps')
            new_tokenized_text = text
            return AttackResult( 
                original_tokenized_text, 
                new_tokenized_text, 
                original_label,
                new_text_label
            )

        return candidate_words

class Hook:
    def __init__(self, module, backward=False):
        if backward:
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_hook(self.hook_fn)
            
    def hook_fn(self, module, input, output):
        self.input = [x.to(get_device()) for x in input]
        self.output = [x.to(get_device()) for x in output]
        
    def close(self):
        self.hook.remove()