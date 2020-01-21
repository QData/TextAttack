from textattack.datasets import TextAttackDataset
from textattack.shared.tokenized_text import TokenizedText

class TextAttackEntailmentDataset(TextAttackDataset):
    """ A generic class for loading entailment data. 
    
    Labels:
        0 - Entailment
        1 - Neutral
        2 - Contradiction
    """
    
    def map_label_str(self, label_str):
        if label_str == 'entailment':
            return 0
        elif label_str == 'neutral':
            return 1
        elif label_str == 'contradiction':
            return 2
        else:
            raise ValueError(f'Unknown SNLI label {label_str}')
    
    def __next__(self):
        if self.i >= len(self.raw_lines):
            raise StopIteration
        line = self.raw_lines[self.i].strip()
        label, premise, hypothesis = line.split('\t')
        try:
            label = int(label)
        except ValueError:
            # If the label is not an integer, it's a label description.
            label = self.map_label_str(label)
        text = TokenizedText.SPLIT_TOKEN.join([premise, hypothesis])
        self.i += 1
        return (label, text)
