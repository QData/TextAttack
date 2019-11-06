from textattack import utils as utils
from textattack.datasets import TextAttackDataset

class YelpSentiment(TextAttackDataset):
    """
    Loads the Yelp Sentiment dataset.
    
    Labels:
        0 - Negative
        1 - Positive

    Args:
        n (int): The number of examples to load
    
    """
    DATA_PATH = '/p/qdata/jm8wx/research_OLD/TextFooler/data/yelp'
    def __init__(self, n=None):
        """ Loads a full dataset from disk. """
        utils.download_if_needed(YelpSentiment.DATA_PATH)
        self.examples = self._load_text_file(YelpSentiment.DATA_PATH, n=n)
        print('YelpSentiment loaded', len(self.examples), 'examples.')
