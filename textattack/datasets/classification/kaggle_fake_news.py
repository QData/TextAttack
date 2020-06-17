from .classification_dataset import ClassificationDataset


class KaggleFakeNews(ClassificationDataset):
    """
    Loads samples from the Kaggle Fake News dataset. https://www.kaggle.com/mrisdal/fake-news
    
    Labels
        0: Real Article
        1: Fake Article

    Args:
        offset (int): line to start reading from
        shuffle (bool): If True, randomly shuffle loaded data
    
    """

    DATA_PATH = "datasets/classification/fake"

    def __init__(self, offset=0, shuffle=False):
        """ Loads a full dataset from disk. """
        self._load_classification_text_file(
            KaggleFakeNews.DATA_PATH, offset=offset, shuffle=shuffle
        )
