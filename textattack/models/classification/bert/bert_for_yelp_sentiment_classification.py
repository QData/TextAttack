from textattack.models.helpers import BERTForClassification

class BERTForYelpSentimentClassification(BERTForClassification):
    """ 
    BERT fine-tuned on the Yelp Sentiment dataset for sentiment classification.
    """
    
    #MODEL_PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/AttackGeneration/models/bert/models/bert-vanilla'
    
    def __init__(self, max_seq_length=32):
        utils.download_if_needed(BertForYelpSentimentClassification.MODEL_PATH)
        self.model = BertForSequenceClassification.from_pretrained(
            BertForYelpSentimentClassification.MODEL_PATH, 
            num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained(
            BertForYelpSentimentClassification.MODEL_PATH)
        self.model.to(utils.get_device())
        self.model.eval()
        self.max_seq_length = max_seq_length
    
    def convert_text_to_ids(self, input_text):
        """ 
        Takes a string input, tokenizes, formats,
        and returns a tensor with text IDs. 
        
        Args:
            input_text (str): The text to tokenize

        Returns:
            The ID of the tokenized text
        
        """
        tokens = self.tokenizer.tokenize(input_text)
        while len(tokens) > self.max_seq_length:
            tokens.pop()
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        while len(ids) < self.max_seq_length + 2:
            ids = ids + [0] # @TODO Is it correct to just pad with zeros?
        return ids
    
    def __call__(self, text_ids):
        if not isinstance(text_ids, torch.Tensor):
            raise ValueError(f'Object of type {type(text_ids)} must be of type torch.tensor')
        with torch.no_grad():
            pred = self.model(text_ids)
        return pred[0]

    def __str__(self):
        return "BERT for Yelp Sentiment Classification"

    MODEL_PATH_CASED ='/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/yelp-cased-2019-11-11-21:00'
    MODEL_PATH_UNCASED = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/yelp-uncased-2019-11-11-11:26'
    def __init__(self, cased=False):
        if cased:
            path = BERTForYelpSentimentClassification.MODEL_PATH_CASED
        else:
            path = BERTForYelpSentimentClassification.MODEL_PATH_UNCASED
        super().__init__(path)
