import gluonnlp
from textattack.datasets import TextAttackDataset

class NewsTest2013EnglishToGerman(TextAttackDataset):
    """ Loads samples from newstest2013 dataset from the publicly available
            WMT2016 translation task. (This is from the 'news' portion of 
            WMT2016. See http://www.statmt.org/wmt16/ for details.)
            
        Samples are loaded as (sample, translation) tuples of string pairs.
            Relies on the GluonNLP library for dataset loading.
    
    Labels:
        0 - Negative
        1 - Positive

    Args:
        src_lang (str): source language
        offset (int): example to start
    
    """
    DATA_PATH = 'datasets/classification/imdb.txt'
    def __init__(self, offset=0):
        examples = gluonnlp.data.WMT2016(segment='newstest2013', src_lang='en', tgt_lang='de')
        # Then account for the offset.
        self.i = offset
        self.examples = examples