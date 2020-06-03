from textattack.datasets import TextAttackDataset
from textattack.shared import TokenizedText

class EntailmentDataset(TextAttackDataset):
    """ 
    A generic class for loading entailment data. 
    
    Labels
        0: Entailment
        1: Neutral
        2: Contradiction
    """
    
    def map_label_str(self, label_str):
        if label_str == 'entailment':
            return 0
        elif label_str == 'neutral':
            return 1
        elif label_str == 'contradiction':
            return 2
        else:
            raise ValueError(f'Unknown entailment label {label_str}')
    
    def _process_example_from_file(self, raw_line):
        line = raw_line.strip()
        label, premise, hypothesis = line.split('\t')
        try:
            label = int(label)
        except ValueError:
            # If the label is not an integer, it's a label description.
            label = self.map_label_str(label)
        text = TokenizedText.SPLIT_TOKEN.join([premise, hypothesis])
        return (text, label)
