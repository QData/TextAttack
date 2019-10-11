

class TextAttackDataset:
    """ A dataset for text attacks.
    
        Any iterable of (label, text_input) pairs qualifies as 
        a TextAttackDataset.
    """
    def __init__(self):
        """ Loads a full dataset from disk. """
        raise NotImplementedException()
    
    def __iter__(self):
        """ Called to iterate through a dataset. """
        raise NotImplementedException()
    
    #@TODO do we need this? :)
    #def __getitem__(self):
     #   raise NotImplementedException()