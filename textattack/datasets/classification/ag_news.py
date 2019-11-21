from textattack import utils as utils
from textattack.datasets import TextAttackDataset

class AGNews(TextAttackDataset):
    """
    Loads the AG News Dataset.
    
    AG is a collection of more than 1 million news articles. News articles have 
    been gathered from more than 2000  news sources by ComeToMyHead in more than 
    1 year of activity. ComeToMyHead is an academic news search engine which has 
    been running since July, 2004. The dataset is provided by the academic 
    community for research purposes in data mining (clustering, classification, 
    etc), information retrieval (ranking, search, etc), xml, data compression, 
    data streaming, and any other non-commercial activity. For more information, 
    please refer to the link 
    http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .

    The AG's news topic classification dataset is constructed by Xiang Zhang 
    (xiang.zhang@nyu.edu) from the dataset above. It is used as a text 
    classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, 
    Yann LeCun. Character-level Convolutional Networks for Text Classification. 
    Advances in Neural Information Processing Systems 28 (NIPS 2015).
    
    Labels:
        0 - World
        1 - Sports
        2 - Business
        3 - Sci/Tech

    Args:
        n (int): The number of examples to load
    
    """
    DATA_PATH = '/p/qdata/jm8wx/research_OLD/TextFooler/data/ag'
    def __init__(self, *args):
        """ Loads a full dataset from disk. """
        super().__init__(*args)
        utils.download_if_needed(AGNews.DATA_PATH)
        self.examples = self._load_text_file(AGNews.DATA_PATH, n=n)
        print('AGNews loaded', len(self.examples), 'examples.')
