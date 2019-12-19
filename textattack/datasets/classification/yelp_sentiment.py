from textattack import utils as utils
from textattack.datasets import TextAttackDataset

class YelpSentiment(TextAttackDataset):
    """
    Loads samples from the Yelp Sentiment dataset.
    
    Labels:
        0 - Negative
        1 - Positive

    Args:
        n (int): The number of examples to load
        offset (int): line to start reading from
    
    """
    DATA_PATH = '/p/qdata/edl9cy/datasets/yelp_sentiment.txt'
    def __init__(self, offset=0):
        """ Loads a full dataset from disk. """
        utils.download_if_needed(YelpSentiment.DATA_PATH)
        self._load_text_file(YelpSentiment.DATA_PATH, offset=offset)

    def _clean_example(self, ex):
        # Removes \\n and \" from the Yelp dataset.
        return ex.replace('\\n',' ').replace('\\"','"')
        
