from textattack.models.helpers import T5ForTextToText

class T5Summarization(T5ForTextToText):
    """ 
    A T5 model trained to summarize English text. Trained on the CNN/Daily 
    Mail summarization dataset.
    
    For more information, please see the T5 paper, "Exploring the Limits of 
    Transfer Learning with a Unified Text-to-Text Transformer".
    Appendix D contains information about the various tasks supported
    by T5.
    """
    def __init__(self, **kwargs):
        super().__init__('summarization', **kwargs)
