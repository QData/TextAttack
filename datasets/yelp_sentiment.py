import utils
from dataset import TextAttackDataset

class YelpSentiment(TextAttackDataset):
    DATA_PATH = '/p/qdata/jm8wx/research/OLD/TextFooler/data/yelp'
    def __init__(self, n=None):
        """ Loads a full dataset from disk. """
        utils.download_if_needed(YelpSentiment.DATA_PATH)
        self.examples = self._load_text_file(YelpSentiment.DATA_PATH, n=n)
        print('YelpSentiment loaded', len(self.examples), 'examples...')