from textattack.attacks import AttackResult, FailedAttackResult
from textattack.attacks.whitebox import WhiteBoxAttack
from textattack.transformations import WordGradientProjection
import torch
# !import code; code.interact(local=vars())
from textattack.utils import get_device

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, sinput, output):
        self.input = input
        self.output = output
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
            id_to_word_converter (func): a function that converts an ID from
                the model's vocabulary to a word (string).
            max_swaps (int): The maximum number of words to swap before giving
                up. Called `r` in the HotFlip paper.
    """
    def __init__(self, model, max_swaps=32):
        if not hasattr(model, 'word_embeddings'):
            raise ValueError('Model needs word embedding matrix for gradient-based word swap')
        if not hasattr(model, 'lookup_table'):
            raise ValueError('Model needs lookup table for gradient-based word swap')
        if not hasattr(model, 'zero_grad'):
            raise ValueError('Model needs `zero_grad()` for gradient-based word swap')
        if not hasattr(model, 'convert_id_to_word'):
            raise ValueError('Model needs `convert_id_to_word()` for gradient-based word swap')
        super().__init__(model)
        self.max_swaps = max_swaps
        # @TODO optionally take other loss functions as a param.
        self.loss = torch.nn.CrossEntropyLoss()
        # @TODO move projection into its own class
        # @TODO call self.get_transformations() so constraints are applied
        # @TODO make recipe file for hotflip
        # self.projection = WordGradientProjection()
        
    def _attack_one(self, original_label, original_tokenized_text):
        # V = word vocab size
        # d = word embedding dimension
        # L = sentence length

        # new_tokenized_text = original_tokenized_text
        text = original_tokenized_text
        y_true = torch.Tensor([original_label]).long().to(get_device())
        swapped_idxs = []
        swaps = 0
        while swaps < self.max_swaps:
            swaps += 1
            # get nonzero word indices, since x is padded with zeros
            nonzero_word_idxs = [word_id.item() for word_id in 
                torch.Tensor(text.ids).nonzero()]
    
            # Word Embedding Lookup Table (Vxd tensor)
            lookup_table = self.model.lookup_table
    
            # set backward hook on the word embeddings for input x
            emb_hook = Hook(self.model.word_embeddings, backward=True)
    
            self.model.zero_grad()
            predictions = self._call_model(text)
            loss = self.loss(predictions, y_true)
            loss.backward()
    
            # grad w.r.t to word embeddings
            import pdb; pdb.set_trace()
            emb_grad = emb_hook.output[0].squeeze()
    
            # grad differences between all flips and original word (eq. 1 from paper)
            diffs = torch.zeros(len(nonzero_word_idxs),lookup_table.size(0))
            for word_idx in nonzero_word_idxs:
                # Don't swap out a word that's already been swapped.
                if word_idx in swapped_idxs:
                    continue
                # pretty sure this is the right way to get the grad w.r.t 
                # the one-hot index of the word, but not completely sure
                b_grads = emb_grad[word_idx].view(1,-1).mm(lookup_table.transpose(0,1)).squeeze()
                a_grad = b_grads[text.ids[word_idx]]
                diffs[word_idx] = b_grads-a_grad
    
            all_words_max_vals, all_words_max_flips = diffs.max(1)
            max_diff, max_word_idx = all_words_max_vals.max(0)
            max_word_flip = all_words_max_flips[max_word_idx]
            swapped_idxs.append(max_word_idx)
    
            max_word_idx = max_word_idx.item() # the index of the word we should flip from x_tensor
            max_word_flip = max_word_flip.item() # the word to flip max_word_idx to 
            
            new_token = self.model.convert_id_to_word(max_word_flip)
            
            try:
                text = text.replace_token_at_index(max_word_idx, new_token)
            except IndexError:
                import pdb; pdb.set_trace()
            
            scores = self._call_model(text).squeeze()
            print('\t scores:', scores)
            new_text_label = scores.argmax().item()
            # print('new_text:', new_text, 'new_text_label:', new_text_label)
            if new_text_label != original_label:
                print('hotflip succeeded after', swaps, 'swaps')
                new_tokenized_text = text
                return AttackResult( 
                    original_tokenized_text, 
                    new_tokenized_text, 
                    original_label,
                    new_text_label
                )
                
        print('Failed with changed text:', text, 'and score:', 
            self._call_model(text).squeeze())
        return FailedAttackResult(original_tokenized_text, original_label)