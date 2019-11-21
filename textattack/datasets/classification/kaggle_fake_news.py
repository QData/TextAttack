from textattack import utils as utils
from textattack.datasets import TextAttackDataset

class KaggleFakeNews(TextAttackDataset):
    """
    Loads the Kaggle Fake News dataset. https://www.kaggle.com/mrisdal/fake-news
    
    Labels:
        0 - Real Article
        1 - Fake Article

    Args:
        n (int): The number of examples to load
        offset (int): line to start reading from
    
    """
    DATA_PATH = '/p/qdata/jm8wx/research_OLD/TextFooler/data/fake'
    def __init__(self):
        """ Loads a full dataset from disk. """
        utils.download_if_needed(KaggleFakeNews.DATA_PATH)
        self.examples = self._load_text_file(KaggleFakeNews.DATA_PATH, n=n, offset=offset)
        print('KaggleFakeNews loaded', len(self.examples), 'examples.')
