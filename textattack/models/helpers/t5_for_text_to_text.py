"""
T5 model trained to generate text from text
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""


import torch
import transformers

from textattack.models.tokenizers import T5Tokenizer
from textattack.shared import utils


class T5ForTextToText(torch.nn.Module):
    """A T5 model trained to generate text from text.

    For more information, please see the T5 paper, "Exploring the Limits of
    Transfer Learning with a Unified Text-to-Text Transformer".
    Appendix D contains information about the various tasks supported
    by T5.

    For usage information, see HuggingFace Transformers documentation section
    on text-to-text with T5:
    https://huggingface.co/transformers/usage.html.

    Args:
        mode (string): Name of the T5 model to use.
        max_length (int): The max length of the sequence to be generated.
            Between 1 and infinity.
        num_beams (int): Number of beams for beam search. Must be between 1 and
            infinity. 1 means no beam search.
        early_stopping (bool): if set to `True` beam search is stopped when at
            least `num_beams` sentences finished per batch. Defaults to `True`.
    """

    def __init__(
        self, mode="english_to_german", max_length=20, num_beams=1, early_stopping=True
    ):
        super().__init__()
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        self.model.to(utils.device)
        self.model.eval()
        self.tokenizer = T5Tokenizer(mode)
        self.max_length = max_length
        self.num_beams = num_beams
        self.early_stopping = early_stopping

    def __call__(self, *args, **kwargs):
        # Generate IDs from the model.
        output_ids_list = self.model.generate(
            *args,
            **kwargs,
            max_length=self.max_length,
            num_beams=self.num_beams,
            early_stopping=self.early_stopping
        )
        # Convert ID tensor to string and return.
        return [self.tokenizer.decode(ids) for ids in output_ids_list]

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
