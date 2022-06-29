"""
HuggingFace Model Wrapper
--------------------------
"""

import torch

from .huggingface_model_wrapper import HuggingFaceModelWrapper

torch.cuda.empty_cache()


class HuggingFaceQAModelWrapper(HuggingFaceModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """

        outputs = []

        for item in text_input_list:

            inputs_dict = self.tokenizer(
                item[1],  # question
                item[0],  # context
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = inputs_dict["input_ids"].tolist()[0]
            model_device = next(self.model.parameters()).device
            inputs_dict.to(model_device)

            with torch.no_grad():
                sub_output = self.model(**inputs_dict)

            answer_start_scores = sub_output.start_logits
            answer_end_scores = sub_output.end_logits
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            outputs.append(
                self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(
                        input_ids[answer_start:answer_end]
                    )
                )
            )

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits
