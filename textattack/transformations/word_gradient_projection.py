import torch

class WordGradientProjection:
    """ Takes a sentence and a gradient and swaps a single word in the direction
        of the gradient.
    """
    def __init__(self):
        # TODO: probably wanna do something here
        pass
    
    def __call__(self, tokenize_text, gradient=None):
        if gradient is None:
            raise ValueError('Cannot project gradient that is None')
        elif not isinstance(gradient, torch.Tensor):
            raise TypeError(f'Cannot project gradient of type{type(gradient)}')
        
        # 
        # project gradient onto tokenized text and return ALL possible 
        #       projections (not just one),
        #
        #   -- should be a list of TokenizedText objects
        # 
        # TokenizedText objects have a tensor .ids and a string
        # .text and know which words they have, a list .raw_words
        
        return [new_tokenize_text1, new_tokenize_text2] # return whatever list u have
                                                            # after projection
    