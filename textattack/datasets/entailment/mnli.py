from .entailment_dataset import EntailmentDataset

class MNLI(EntailmentDataset):
    """
    Loads samples from the MNLI dataset. The *mismatched* examples come from a 
    distribution different from the one seen at training time.
    
    See https://www.nyu.edu/projects/bowman/multinli/paper.pdf for more details.
    
    Labels
        0: Entailment
        1: Neutral
        2: Contradiction

    Args:
        offset (int): line to start reading from
        mismatched (bool): whether to use mismatched dataset. Defaults to false.
    """
    MATCHED_DATA_PATH = 'datasets/entailment/mnli_matched'
    MISMATCHED_DATA_PATH = 'datasets/entailment/mnli_mismatched'
    def __init__(self, offset=0, mismatched=False):
        """ Loads a full dataset from disk. """
        path = MNLI.MISMATCHED_DATA_PATH if mismatched else MNLI.MATCHED_DATA_PATH
        self._load_classification_text_file(path, offset=offset)
