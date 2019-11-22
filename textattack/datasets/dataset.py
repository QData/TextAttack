class TextAttackDataset:
    """
    A dataset for text attacks.
    
    Any iterable of (label, text_input) pairs qualifies as 
    a TextAttackDataset.
    
    """
    def __init__(self):
        """ Loads a full dataset from disk. Typically stores tuples in
            `self.examples`.
        """
        raise NotImplementedError()
    
    def __iter__(self):
        return self.examples.__iter__()
    
    def __next__(self):
        return self.examples.__next__()
    
    def _load_text_file(self, text_file_name, n=None, offset=0):
        """ Loads (label, text) pairs from a text file. 
        
            Format must look like:
            
                1 this is a great little ...
                0 "i love hot n juicy .  ...
                0 "\""this world needs a ...
            
            Arguments:
                n (int): number of samples to return
                offset (int): line to start reading from
        """
        examples = []
        text_file = open(text_file_name, 'r')
        i = 0
        for j, raw_line in enumerate(text_file.readlines()):
            if j < offset: 
                continue
            tokens = raw_line.strip().split()
            label = int(tokens[0])
            text = ' '.join(tokens[1:])
            examples.append((label, text))
            i += 1
            if n and i >= n: break
        return examples
