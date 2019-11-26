from textattack.attacks import AttackResult, FailedAttackResult
from textattack.attacks.whitebox import WhiteBoxAttack
from textattack.transformations import WordGradientProjection
import torch
from textattack.utils import get_device


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
        self.projection = WordGradientProjection()

        # @TODO move projection into its own class
        # @TODO call self.get_transformations() so constraints are applied
        # @TODO add word-embedding-distance constraint for hotflip
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
        import pdb; pdb.set_trace()
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
            emb_grad = emb_hook.output[0].to(get_device()).squeeze()
    
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
        """
        y_true = torch.Tensor([original_label]).long()
        x_tensor = torch.Tensor(original_tokenized_text.ids).view(1,-1).long()

        # get nonzero word indices, since x is padded with zeros
        token_ids_tensor = torch.Tensor(original_tokenized_text.ids)
        nonzero_word_idxs = (token_ids_tensor != self.pad_index).nonzero().view(-1)
        
        if self.model_type == 'bert':
            #remove BOS and EOS
            nonzero_word_idxs = nonzero_word_idxs[1:-1]
        
        
        # Word Embedding Lookup Table (Vxd tensor)
        lookup_table = self.model.emb_layer.embedding.weight.data
        self.model.emb_layer.embedding.weight.requires_grad = True


        swapped_indices = []
        swaps = 0
        while swaps < self.max_swaps:
            # set backward hook on the word embeddings for input x
            emb_hook = Hook(self.model.emb_layer.embedding,backward=True)

            self.model.zero_grad()
            predictions = self.model(x_tensor)

            loss = self.loss(predictions,y_true)
            loss.backward()

            # grad w.r.t to word embeddings
            emb_grads = emb_hook.output[0]

            # grad differences between all flips and original word (eq. 1 from paper)
            diffs = torch.zeros(len(nonzero_word_idxs),lookup_table.size(0))

            for word_idx in nonzero_word_idxs:
                # get the grad w.r.t the one-hot index of the word
                b_grads = emb_grads[word_idx].view(1,-1).mm(lookup_table.transpose(0,1)).squeeze()
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
        """




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
        lookup_table = self.model.emb_layer.embedding.word_embeddings.weight.data

        swapped_indices = []
        swaps = 0
        while swaps < self.max_swaps:
            # set backward hook on the word embeddings for input x
            emb_hook = Hook(self.model.emb_layer.embedding.word_embeddings,backward=True)

            self.model.zero_grad()
            predictions = self.model(x_tensor)

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
