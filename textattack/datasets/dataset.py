from textattack.shared import utils

class TextAttackDataset:
    """
    A dataset for text attacks.
    
    Any iterable of (label, text_input) pairs qualifies as 
    a TextAttackDataset.
    
    """
    def __init__(self):
        """ Loads a full dataset from disk. """
        raise NotImplementedError()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i >= len(self.raw_lines):
            raise StopIteration
        tokens = self.raw_lines[self.i].strip().split()
        label = int(tokens[0])
        text = ' '.join(tokens[1:])
        self.i += 1
        return (label, text)
    
    def _load_text_file(self, text_file_name, offset=0):
        """ Loads (label, text) pairs from a text file. 
        
            Format must look like:
            
                1 this is a great little ...
                0 "i love hot n juicy .  ...
                0 "\""this world needs a ...
            
            Arguments:
                n (int): number of samples to return
                offset (int): line to start reading from
        """
        text_file_path = utils.download_if_needed(text_file_name)
        text_file = open(text_file_path, 'r')
        raw_lines = text_file.readlines()[offset:]
        self.raw_lines = [self._clean_example(ex) for ex in raw_lines]
        self.i = 0
        text_file.close()
    
    def _clean_example(self, ex):
        """ Optionally pre-processes an input string before some tokenization.
            Only necessary for some datasets. """
        return ex
