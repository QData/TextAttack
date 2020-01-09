from textattack import utils as utils
from textattack.datasets import TextAttackDataset
from textattack.tokenized_text import TokenizedText

class SNLI(TextAttackDataset):
    """
    Loads samples from the SNLI dataset.
    
    Labels:
        0 - Entailment
        1 - Neutral
        2 - Contradiction

    Args:
        offset (int): line to start reading from
    
    """
    DATA_PATH = '/p/qdata/jm8wx/research_OLD/textfooler/data/snli-uncased'
    def __init__(self, offset=0):
        """ Loads a full dataset from disk. """
        self._load_text_file(SNLI.DATA_PATH, offset=offset)
    
    def map_label_str(self, label_str):
        if label == 'entailment:
            return 0
        elif label == 'neutral':
            return 1
        elif label == 'contradiction':
            return 2
        else:
            raise ValueError(f'Unknown SNLI label {label_str}')
    
    def __next__(self):
        if self.i >= len(self.raw_lines):
            raise StopIteration
        line = self.raw_lines[self.i].strip()
        label, premise, hypothesis = line.split('\t')
        label = int(label)
        text = TokenizedText.SPLIT_TOKEN.join([premise, hypothesis])
        self.i += 1
        return (label, text)
