"""
T5 model trained to generate text from text
---------------------------------------------------------------------

"""
import json
import os

import torch
import transformers

from textattack.model_args import TEXTATTACK_MODELS
from textattack.models.tokenizers import T5Tokenizer


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
        output_max_length (int): The max length of the sequence to be generated.
            Between 1 and infinity.
        input_max_length (int): Max length of the input sequence.
        num_beams (int): Number of beams for beam search. Must be between 1 and
            infinity. 1 means no beam search.
        early_stopping (bool): if set to `True` beam search is stopped when at
            least `num_beams` sentences finished per batch. Defaults to `True`.
    """

    def __init__(
        self,
        mode="english_to_german",
        output_max_length=20,
        input_max_length=64,
        num_beams=1,
        early_stopping=True,
    ):
        super().__init__()
        self.model = transformers.T5ForConditionalGeneration.from_pretrained("t5-base")
        self.model.eval()
        self.tokenizer = T5Tokenizer(mode, max_length=output_max_length)
        self.mode = mode
        self.output_max_length = output_max_length
        self.input_max_length = input_max_length
        self.num_beams = num_beams
        self.early_stopping = early_stopping

    def __call__(self, *args, **kwargs):
        # Generate IDs from the model.
        output_ids_list = self.model.generate(
            *args,
            **kwargs,
            max_length=self.output_max_length,
            num_beams=self.num_beams,
            early_stopping=self.early_stopping,
        )
        # Convert ID tensor to string and return.
        return [self.tokenizer.decode(ids) for ids in output_ids_list]

    def save_pretrained(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        config = {
            "mode": self.mode,
            "output_max_length": self.output_max_length,
            "input_max_length": self.input_max_length,
            "num_beams": self.num_beams,
            "early_stoppping": self.early_stopping,
        }
        # We don't save it as `config.json` b/c that name conflicts with HuggingFace's `config.json`.
        with open(os.path.join(output_dir, "t5-wrapper-config.json"), "w") as f:
            json.dump(config, f)
        self.model.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, name_or_path):
        """Load trained LSTM model by name or from path.

        Args:
            name_or_path (str): Name of the model (e.g. "t5-en-de") or model saved via `save_pretrained`.
        """
        if name_or_path in TEXTATTACK_MODELS:
            t5 = cls(TEXTATTACK_MODELS[name_or_path])
            return t5
        else:
            config_path = os.path.join(name_or_path, "t5-wrapper-config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            t5 = cls.__new__(cls)
            for key in config:
                setattr(t5, key, config[key])
            t5.model = transformers.T5ForConditionalGeneration.from_pretrained(
                name_or_path
            )
            t5.tokenizer = T5Tokenizer(t5.mode, max_length=t5.output_max_length)
            return t5

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
