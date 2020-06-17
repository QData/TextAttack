from .classification_dataset import ClassificationDataset


class YelpSentiment(ClassificationDataset):
    """
    Loads samples from the Yelp Sentiment dataset.
    
    Labels
        0: Negative
        1: Positive

    Args:
        offset (int): line to start reading from
        shuffle (bool): If True, randomly shuffle loaded data
    
    """

    DATA_PATH = "datasets/classification/yelp_sentiment.txt"

    def __init__(self, offset=0, shuffle=False):
        """ Loads a full dataset from disk. """
        self._load_classification_text_file(
            YelpSentiment.DATA_PATH, offset=offset, shuffle=shuffle
        )

    def _clean_example(self, ex):
        """ Applied to every text example loaded from disk. 
            Removes \\n and \" from the Yelp dataset. 
        """
        return ex.replace("\\n", " ").replace('\\"', '"')
