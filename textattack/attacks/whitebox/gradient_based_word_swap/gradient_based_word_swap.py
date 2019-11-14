from textattack.attacks import AttackResult, FailedAttackResult
from textattack.attacks.whitebox import WhiteBoxAttack
from textattack.transformations import WordGradientProjection
import torch
from pdb import set_trace as stop
# !import code; code.interact(local=vars())

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
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
    """
    #
    # decided to actually not call the attack HotFlip cuz it's more generic
    # than that
    #
    def __init__(self, model, max_swaps=1):
        super().__init__(model)
        #
        # this is the max # of swaps to try before giving up
        #  it's called r' in the Hotflip paper
        self.max_swaps = max_swaps
        self.loss = torch.nn.CrossEntropyLoss()
        self.projection = WordGradientProjection()
        
    def _attack_one(self, original_label, original_tokenized_text):
        # V = word vocab size
        # d = word embedding dimension
        # L = sentence length

        # new_tokenized_text = original_tokenized_text
        # swaps = 0
        # while swaps < self.max_swaps:

        y_true = torch.Tensor(original_label).long()
        x_tensor = torch.Tensor(original_tokenized_text.ids).view(1,-1).long()

        # get nonzero word indices, since x is padded with zeros
        nonzero_word_idxs = [word_id.item() for word_id in 
            torch.Tensor(original_tokenized_text.ids).nonzero()]

        # Word Embedding Lookup Table (Vxd tensor)
        lookup_table = self.model.model.bert.embeddings.word_embeddings.weight.data

        # set backward hook on the word embeddings for input x
        emb_hook = Hook(self.model.model.bert.embeddings.word_embeddings,backward=True)

        self.model.model.zero_grad()
        predictions = self.model.model(x_tensor)
        loss = self.loss(predictions[0],y_true)
        loss.backward()

        # grad w.r.t to word embeddings
        emb_grad = emb_hook.output[0].squeeze()

        # grad differences between all flips and original word (eq. 1 from paper)
        diffs = torch.zeros(len(nonzero_word_idxs),lookup_table.size(0))
        for word_idx in nonzero_word_idxs:
            # pretty sure this is the right way to get the grad w.r.t 
            # the one-hot index of the word, but not completely sure
            b_grads = emb_grad[word_idx].view(1,-1).mm(lookup_table.transpose(0,1)).squeeze()
            a_grad = b_grads[original_tokenized_text.ids[word_idx]]
            diffs[word_idx] = b_grads-a_grad

        all_words_max_vals,all_words_max_flips = diffs.max(1)
        max_diff,max_word_idx = all_words_max_vals.max(0)
        max_word_flip = all_words_max_flips[max_word_idx]

        max_word_idx = max_word_idx.item() # the index of the word we should flip from x_tensor
        max_word_flip = max_word_flip.item() # the word to flip max_word_idx to 

        # diff_max_idx_flat = diffs.view(1, -1).argmax(1)
        # diff_max_idx = torch.cat(((diff_max_idx_flat/diffs.size(1)).view(-1, 1),
        #                 (diff_max_idx_flat % diffs.size(1)).view(-1, 1)),dim=1)[0]

        # new_text_options = self.get_transformations(self.projection, 
        #     new_tokenized_text, gradient=gradient)
        # # If we couldn't find any next sentence that meets the constraints,
        # # break cuz we failed
        # if not len(transformations):
        #     break
        # #
        # # "We choose the vector with biggest increase in loss:"
        # #   -- do something like this to pick the best one
        # # 
        # scores = self.model(new_text_options)
        # best_index = scores[:, original_label].argmin()
        # new_tokenized_text = new_text_options[best_index]
        # #
        # # check if the label changed -- if we did, stop and return
        # # successful result
        # #
        # new_text_label = scores[best_index].argmax().item()
        # if new_text_label != original_label:
        #     return AttackResult( 
        #         original_tokenized_text, 
        #         new_tokenized_text, 
        #         original_label,
        #         new_text_label
        #     )
        # 
        # if it didnt change yet, increase # swaps and keep trying
        #
        # swaps += 1
        # if we get here, we failed cuz swaps == self.max_swaps
        return FailedAttackResult(original_tokenized_text, original_label)