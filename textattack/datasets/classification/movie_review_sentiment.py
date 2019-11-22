from textattack import utils as utils
from textattack.datasets import TextAttackDataset

class MovieReviewSentiment(TextAttackDataset):
    """
    Loads samples from the Movie Review Dataset. The "MR" dataset is comprised 
    of sentence-level sentiment classification on positive and negative movie 
    reviews (Pang and Lee, 2005).
    
    Labels:
        0 - Negative
        1 - Positive

    Args:
        n (int): The number of examples to load
        offset (int): line to start reading from
    
    """
    DATA_PATH = '/p/qdata/jm8wx/research/text_attacks/textattack_data/mr.txt'
    def __init__(self, n=None, offset=None):
        """ Loads a full dataset from disk. """
        utils.download_if_needed(MovieReviewSentiment.DATA_PATH)
        self.examples = self._load_text_file(MovieReviewSentiment.DATA_PATH, n=n,
            offset=offset)
        print('MovieReviewSentiment loaded', len(self.examples), 'examples.')
