from textattack.models.wrappers import ModelWrapper


class HuggingFaceLLMWrapper(ModelWrapper):
    """A wrapper around HuggingFace for LLMs.

    Args:
        model: A HuggingFace pretrained LLM
        tokenizer: A HuggingFace pretrained tokenizer
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list):
        """Returns a list of responses to the given input list."""
        model_device = next(self.model.parameters()).device
        input_ids = self.tokenizer(text_input_list, return_tensors="pt").input_ids
        input_ids.to(model_device)

        outputs = self.model.generate(
            input_ids, max_new_tokens=512, pad_token_id=self.tokenizer.eos_token_id
        )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if len(text_input_list) == 1:
            return responses[0]
        return responses
