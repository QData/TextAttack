import pandas as pd

from .model_wrapper import ModelWrapper


class SklearnModelWrapper(ModelWrapper):
    """Loads a scikit-learn model and tokenizer (tokenizer implements
    `transform` and model implements `predict_proba`).

    May need to be extended and modified for different types of
    tokenizers.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list):
        encoded_text_matrix = self.tokenizer.transform(text_input_list).toarray()
        tokenized_text_df = pd.DataFrame(
            encoded_text_matrix, columns=self.tokenizer.get_feature_names()
        )
        return self.model.predict_proba(tokenized_text_df)
