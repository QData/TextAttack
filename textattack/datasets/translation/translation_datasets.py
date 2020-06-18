from textattack.datasets import TextAttackDataset


class NewsTest2013EnglishToGerman(TextAttackDataset):
    """ 
    Loads samples from newstest2013 dataset from the publicly available
    WMT2016 translation task. (This is from the 'news' portion of 
    WMT2016. See http://www.statmt.org/wmt16/ for details.) Dataset
    sourced from GluonNLP library.
        
    Samples are loaded as (input, translation) tuples of string pairs.
    
    Args:
        offset (int): line to start reading from
        shuffle (bool): If True, randomly shuffle loaded data

    """

    DATA_PATH = "datasets/translation/NewsTest2013EnglishToGerman"

    def __init__(self, offset=0, shuffle=False):
        self._load_pickle_file(NewsTest2013EnglishToGerman.DATA_PATH, offset=offset)
        if shuffle:
            self._shuffle_data()
