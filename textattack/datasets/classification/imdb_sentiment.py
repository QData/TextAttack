from .classification_dataset import ClassificationDataset


class IMDBSentiment(ClassificationDataset):
    """
    Loads samples from the IMDB Movie Review Sentiment dataset.
    
    Labels
        0: Negative
        1: Positive

    Args:
        offset (int): line to start reading from
        shuffle (bool): If True, randomly shuffle loaded data
    
    """

    DATA_PATH = "datasets/classification/imdb.txt"

    def __init__(self, offset=0, shuffle=False):
        """ Loads a full dataset from disk. """
        self._load_classification_text_file(
            IMDBSentiment.DATA_PATH, offset=offset, shuffle=shuffle
        )
