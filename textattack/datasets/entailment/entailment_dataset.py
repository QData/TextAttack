import collections

from textattack.datasets import TextAttackDataset
from textattack.shared import AttackedText


class EntailmentDataset(TextAttackDataset):
    """ 
    A generic class for loading entailment data. 
    
    Labels
        0: Entailment
        1: Neutral
        2: Contradiction
    """

    def _label_str_to_int(self, label_str):
        if label_str == "entailment":
            return 0
        elif label_str == "neutral":
            return 1
        elif label_str == "contradiction":
            return 2
        else:
            raise ValueError(f"Unknown entailment label {label_str}")

    def _process_example_from_file(self, raw_line):
        line = raw_line.strip()
        label, premise, hypothesis = line.split("\t")
        try:
            label = int(label)
        except ValueError:
            # If the label is not an integer, it's a label description.
            label = self._label_str_to_int(label)
        text_input = collections.OrderedDict(
            [("premise", premise), ("hypothesis", hypothesis),]
        )
        return (text_input, label)
