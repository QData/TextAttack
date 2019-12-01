import nltk
import torch

from textattack.attacks import AttackResult, FailedAttackResult
from textattack.attacks.whitebox import WhiteBoxAttack
from textattack.transformations import WordGradientProjection
from textattack.utils import get_device

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
            
    def hook_fn(self, module, input, output):
        self.input = [x.to(get_device()) for x in input]
        self.output = [x.to(get_device()) for x in output]
        
    def close(self):
        self.hook.remove()

class GradientBasedWordSwap(WhiteBoxAttack):
    """ Uses the model's gradient to iteratively replace words until
        a model's prediction score changes. 
        
        Based off of HotFlip: White-Box Adversarial Examples for Text 
            Classification (Ebrahimi et al., 2018).
        
            https://arxiv.org/pdf/1712.06751.pdf
        
        Arguments:
            model (nn.Module): The model to attack. Model must have a 
                `word_embeddings` matrix and `convert_id_to_word` function.
            max_swaps (int): The maximum number of words to swap before giving
                up. Called `r` in the HotFlip paper.
    """
    def __init__(self, model, replace_stopwords=True, max_swaps=32):
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
        if replace_stopwords:
            self.stopwords = set(nltk.corpus.stopwords.words('english'))
        else:
            self.stopwords = set()
        
    def _attack_one(self, original_label, original_tokenized_text):
        # V = word vocab size
        # d = word embedding dimension
        # L = sentence length
            
        # Word Embedding Lookup Table (Vxd tensor)
        lookup_table = self.model.lookup_table.to(get_device())
        lookup_table_transpose = lookup_table.transpose(0,1)

        # new_tokenized_text = original_tokenized_text
        text = original_tokenized_text
        y_true = torch.Tensor([original_label]).long().to(get_device())
        swapped_idxs = []
        swaps = 0
        
        # get nonzero word indices, since x is padded with zeros
        nonzero_token_idxs = [i for i, word_id in enumerate(text.ids) 
            if (word_id != self.pad_id) and (word_id != self.oov_id)]
        
        # get indices of words that aren't stopwords
        swappable_token_idxs = []
        for i in nonzero_token_idxs:
            token = text.tokens[i]
            print(i,'/',token, token==self.pad_id)
            print('\t',text.ids[i])
            if token.isalpha() and (token.lower() not in self.stopwords):
                swappable_token_idxs.append(i)
        
        while (swaps < self.max_swaps) and (len(swappable_token_idxs) > 0):
            swaps += 1
    
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
                
        print('Failed with changed text:', text, 'and score:', self._call_model(text).squeeze())
        return FailedAttackResult(original_tokenized_text, original_label)