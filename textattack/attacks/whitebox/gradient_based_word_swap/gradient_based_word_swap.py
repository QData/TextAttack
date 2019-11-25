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

        y_true = torch.Tensor(original_label).long()
        x_tensor = torch.Tensor(original_tokenized_text.ids).view(1,-1).long()


        # get nonzero word indices, since x is padded with zeros
        nonzero_word_idxs = [word_id.item() for word_id in 
            torch.Tensor(original_tokenized_text.ids).nonzero()]
        nonzero_word_idxs = nonzero_word_idxs[1:-1] #remove BOS and EOS
        
        # Word Embedding Lookup Table (Vxd tensor)
        lookup_table = self.model.model.bert.embeddings.word_embeddings.weight.data

        swapped_indices = []
        swaps = 0
        while swaps < self.max_swaps:
            # set backward hook on the word embeddings for input x
            emb_hook = Hook(self.model.model.bert.embeddings.word_embeddings,backward=True)

            self.model.model.zero_grad()
            predictions = self.model.model(x_tensor)

            loss = self.loss(predictions[0],y_true.repeat(predictions[0].size(0)))
            loss.backward()

            # grad w.r.t to word embeddings
            emb_grads = emb_hook.output[0]

            # grad differences between all flips and original word (eq. 1 from paper)
            diffs = torch.zeros(len(nonzero_word_idxs),lookup_table.size(0))

            for word_idx in nonzero_word_idxs:
                # get the grad w.r.t the one-hot index of the word
                b_grads = emb_grads[0][word_idx].view(1,-1).mm(lookup_table.transpose(0,1)).squeeze()
                a_grad = b_grads[original_tokenized_text.ids[word_idx]]
                diffs[word_idx] = b_grads-a_grad

            all_words_max_vals,all_words_max_flips = diffs.max(1)
            ranked_diffs,ranked_words = all_words_max_vals.sort(descending=True)
            max_word_idx = ranked_words[0]
            max_word_flip = all_words_max_flips[max_word_idx]

            max_word_idx = max_word_idx.item() # the index of the word we should flip from x_tensor
            max_word_flip = max_word_flip.item() # the word to flip max_word_idx to

            if max_word_idx not in swapped_indices:
                x_tensor[0][max_word_idx] = max_word_flip
                swapped_indices.append(max_word_idx)


            swaps += 1

        return FailedAttackResult(original_tokenized_text, original_label)




class GradientBasedWordSwapBeam(WhiteBoxAttack):
    """ Uses the model's gradient to iteratively replace words until
        a model's prediction score changes. 
        
        Based off of HotFlip: White-Box Adversarial Examples for Text 
            Classification (Ebrahimi et al., 2018).
        
            https://arxiv.org/pdf/1712.06751.pdf
    """
    def __init__(self, model, max_swaps=1):
        super().__init__(model)
        # this is the max # of swaps to try before giving up
        #  it's called r' in the Hotflip paper
        self.max_swaps = max_swaps
        self.loss = torch.nn.CrossEntropyLoss()
        self.projection = WordGradientProjection()
        self.beam_size = 1
        self.beam = False
        
    def _attack_one(self, original_label, original_tokenized_text):
        # V = word vocab size
        # d = word embedding dimension
        # L = sentence length

        y_true = torch.Tensor(original_label).long()
        x_tensor = torch.Tensor(original_tokenized_text.ids).view(1,-1).long()
        assert x_tensor.size(0) == 1
        # x_tensor = x_tensor.repeat(self.beam_size,1)
        # y_true = y_true.repeat(self.beam_size)


        # get nonzero word indices, since x is padded with zeros
        nonzero_word_idxs = [word_id.item() for word_id in 
            torch.Tensor(original_tokenized_text.ids).nonzero()]
        
        # Word Embedding Lookup Table (Vxd tensor)
        lookup_table = self.model.model.bert.embeddings.word_embeddings.weight.data

        swapped_indices = []
        swaps = 0
        while swaps < self.max_swaps:
            # set backward hook on the word embeddings for input x
            emb_hook = Hook(self.model.model.bert.embeddings.word_embeddings,backward=True)

            self.model.model.zero_grad()
            predictions = self.model.model(x_tensor)

            loss = self.loss(predictions[0],y_true.repeat(predictions[0].size(0)))
            loss.backward()

            # grad w.r.t to word embeddings
            emb_grads = emb_hook.output[0]
            emb_grads = torch.cat([beam_grads.unsqueeze(0) for beam_grads in emb_grads],0)

            # grad differences between all flips and original word (eq. 1 from paper)
            diffs = torch.zeros(self.beam_size,len(nonzero_word_idxs),lookup_table.size(0))
            
            repeated_lt = lookup_table.transpose(0,1).unsqueeze(0).repeat(emb_grads.size(0),1,1)
            for word_idx in nonzero_word_idxs:
                # get the grad w.r.t the one-hot index of the word
                b_grads = emb_grads[:,word_idx].unsqueeze(1).bmm(repeated_lt).squeeze(1)
                a_grad = (b_grads[:,original_tokenized_text.ids[word_idx]]).unsqueeze(1)
                diffs[:,word_idx,:] = b_grads-a_grad

            if swaps == 0: # get initial k beam flips
                all_words_max_vals,all_words_max_flips = diffs[0].max(1)
                ranked_diffs,ranked_words = all_words_max_vals.sort(descending=True)
                indexes_to_flip = ranked_words[0:self.beam_size]
                words_to_flip_to = all_words_max_flips[indexes_to_flip]
                x_tensor = x_tensor.repeat(self.beam_size,1)
                for beam_k in range(self.beam_size):
                    x_tensor[beam_k][indexes_to_flip[beam_k]] = words_to_flip_to[beam_k]
            else:
                candidates = []
                for beam_k in range(self.beam_size):
                    all_words_max_vals,all_words_max_flips = diffs[beam_k].max(1)
                    ranked_diffs,indexes_to_flip = all_words_max_vals.sort(descending=True)
                    words_to_flip_to = all_words_max_flips[indexes_to_flip]
                    for i in range(len(ranked_diffs)):
                        beam_x_tensor = x_tensor[beam_k].clone()
                        beam_x_tensor[indexes_to_flip[i]] = words_to_flip_to[i]
                        candidates.append([beam_x_tensor,ranked_diffs[i].item()])

                top_k = sorted(candidates, key=lambda tup:tup[1])[0:self.beam_size]

                x_tensor = torch.cat([tensor_i[0].view(1,-1) for tensor_i in top_k],0)

            swaps += 1



        return FailedAttackResult(original_tokenized_text, original_label)
