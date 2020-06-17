from textattack.shared import utils

from .classification_dataset import ClassificationDataset


class AGNews(ClassificationDataset):
    """
    Loads samples from the AG News Dataset.
    
    AG is a collection of more than 1 million news articles. News articles have 
    been gathered from more than 2000  news sources by ComeToMyHead in more than 
    1 year of activity. ComeToMyHead is an academic news search engine which has 
    been running since July, 2004. The dataset is provided by the academic 
    community for research purposes in data mining (clustering, classification, 
    etc), information retrieval (ranking, search, etc), xml, data compression, 
    data streaming, and any other non-commercial activity. For more information, 
    please refer to the link 
    http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html.

    The AG's news topic classification dataset was constructed by Xiang Zhang 
    (xiang.zhang@nyu.edu) from the dataset above. It is used as a text 
    classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, 
    Yann LeCun. Character-level Convolutional Networks for Text Classification. 
    Advances in Neural Information Processing Systems 28 (NIPS 2015).
    
    Labels
        0: World
        1: Sports
        2: Business
        3: Sci/Tech

    Args:
        offset (int): line to start reading from
        shuffle (bool): If True, randomly shuffle loaded data
    
    """

    DATA_PATH = "datasets/classification/ag_news.txt"

    def __init__(self, offset=0, shuffle=False):
        """ Loads a full dataset from disk. """
        self._load_classification_text_file(
            AGNews.DATA_PATH, offset=offset, shuffle=shuffle
        )
