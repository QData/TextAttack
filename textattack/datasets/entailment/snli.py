from .entailment_dataset import EntailmentDataset

class SNLI(EntailmentDataset):
    """
    Loads samples from the SNLI dataset.
    
    Labels
        0: Entailment
        1: Neutral
        2: Contradiction

    Args:
        offset (int): line to start reading from
    
    """
    DATA_PATH = 'datasets/entailment/snli'
    def __init__(self, offset=0):
        """ Loads a full dataset from disk. """
        self._load_classification_text_file(SNLI.DATA_PATH, offset=offset)
