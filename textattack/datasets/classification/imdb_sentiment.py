from textattack import utils as utils
from textattack.datasets import TextAttackDataset

class IMDBSentiment(TextAttackDataset):
    """
    Loads samples from the IMDB Movie Review Sentiment dataset.
    
    Labels:
        0 - Negative
        1 - Positive

    Args:
        n (int): The number of examples to load
        offset (int): line to start reading from
    
    """
    DATA_PATH = '/p/qdata/jm8wx/research/text_attacks/textattack_data/imdb.txt'
    def __init__(self, n=None, offset=None):
        """ Loads a full dataset from disk. """
        utils.download_if_needed(IMDBSentiment.DATA_PATH)
        self.examples = self._load_text_file(IMDBSentiment.DATA_PATH, n=n, 
            offset=offset)
        print('IMDBSentiment loaded', len(self.examples), 'examples.')
