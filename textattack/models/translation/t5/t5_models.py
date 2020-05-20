from textattack.models.helpers import T5ForTextToText

class T5EnglishToGerman(T5ForTextToText):
    """ 
    A T5 model trained to translate English text to German. Trained on the 
    same training data as [Vaswani et al., 2017] (i.e. News Commentary
    v13, Common Crawl, Europarl v7) and newstest2013 used as a 
    validation set [Bojar et al., 2014].
    
    For more information, please see the T5 paper, "Exploring the Limits of 
    Transfer Learning with a Unified Text-to-Text Transformer".
    Appendix D contains information about the various tasks supported
    by T5.
    """
    def __init__(self, **kwargs):
        super().__init__('english_to_german', **kwargs)
        
class T5EnglishToFrench(T5ForTextToText):
    """ 
    A T5 model trained to translate English text to French. Trained on the 
    standard training data from 2015 and newstest2014 as a validation 
    set [Bojar et al., 2015].
    
    For more information, please see the T5 paper, "Exploring the Limits of 
    Transfer Learning with a Unified Text-to-Text Transformer".
    Appendix D contains information about the various tasks supported
    by T5.
    """
    def __init__(self, **kwargs):
        super().__init__('english_to_french', **kwargs)
        
class T5EnglishToRomanian(T5ForTextToText):
    """ 
    A T5 model trained to translate English text to Romanian. 
    
    English to Romanian is a standard lower-resource machine translation 
    benchmark. This model was trained using the train and validation 
    sets from WMT 2016 [Bojar et al., 2016].Trained on the standard 
    training data from 2015 and newstest2014 as a validation set 
    [Bojar et al., 2015].

    For more information, please see the T5 paper, "Exploring the Limits of 
    Transfer Learning with a Unified Text-to-Text Transformer".
    Appendix D contains information about the various tasks supported
    by T5.
    """
    def __init__(self, **kwargs):
        super().__init__('english_to_romanian', **kwargs)
