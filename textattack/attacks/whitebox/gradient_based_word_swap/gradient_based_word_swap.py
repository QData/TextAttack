from textattack.attacks import AttackResult, FailedAttackResult
from textattack.attacks.whitebox import WhiteBoxAttack
from textattack.transformations import WordGradientProjection

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
    def __init__(self, model, max_swaps=16):
        super().__init__(model)
        #
        # this is the max # of swaps to try before giving up
        #  it's called r' in the Hotflip paper
        #
        self.max_swaps = max_swaps
        self.projection = WordGradientProjection()
        
    def _attack_one(self, original_label, original_tokenized_text):
        new_tokenized_text = original_tokenized_text
        swaps = 0
        while swaps < self.max_swaps:
            #
            # get the gradient of the text we're working with
            #
            gradient = get_gradient_somehow(self.model)
            #
            # self.transformations is a method in Attack, which is the parent
            # of WhiteBoxAttack, so we can call it here.
            # 
            # self.transformations calls transformation(text) on some text,
            # but it also applies any constraints that have been added to the
            # attack via `self.add_constraints()`
            # 
            # all kwargs get passed to the transformation, so the gradient
            #   will be passed to the projection.__call__ as gradient=gradient
            #
            new_text_options = self.get_transformations(self.projection, 
                new_tokenized_text, gradient=gradient)
            # If we couldn't find any next sentence that meets the constraints,
            # break cuz we failed
            if not len(transformations):
                break
            #
            # "We choose the vector with biggest increase in loss:"
            #   -- do something like this to pick the best one
            # 
            scores = self.model(new_text_options)
            best_index = scores[:, original_label].argmin()
            new_tokenized_text = new_text_options[best_index]
            #
            # check if the label changed -- if we did, stop and return
            # successful result
            #
            new_text_label = scores[best_index].argmax().item()
            if new_text_label != original_label:
                return AttackResult( 
                    original_tokenized_text, 
                    new_tokenized_text, 
                    original_label,
                    new_text_label
                )
            # 
            # if it didnt change yet, increase # swaps and keep trying
            #
            swaps += 1
        # if we get here, we failed cuz swaps == self.max_swaps
        return FailedAttackResult(original_tokenized_text, original_label)