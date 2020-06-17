from textattack.models.tokenizers import AutoTokenizer


class BERTTokenizer(AutoTokenizer):
    """ 
    A generic class that convert text to tokens and tokens to IDs. Intended
    for fine-tuned BERT models.
    """

    def __init__(self, name="bert-base-uncased", max_length=256):
        super().__init__(name, max_length=max_length)
