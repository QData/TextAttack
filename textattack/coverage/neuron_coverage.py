import logging

import torch
import transformers
from tqdm import tqdm
import itertools
import copy

import textattack
from collections import defaultdict
from .coverage import ExtrinsicCoverage

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


COVERAGE_MODEL_TYPES = ["bert", "albert", "distilbert", "roberta"]


class neuronCoverage(ExtrinsicCoverage):
    """
    ``neuronCoverage`` measures the neuron coverage acheived by a testset
    Args:
                    test_model(Union[str, torch.nn.Module]): name of the pretrained language model from `transformers`
                                    or the actual test model as a `torch.nn.Module` class. Default is "bert base uncased" from `transformers`.
                    tokenizer (:obj:``, optional): If `test_model` is not a pretrained model from `transformers, need to provide
                                    the tokenizer here.
                    max_seq_len (int): Maximum sequence length accepted by the model to be tested.  However, if you are using a pretrained model from `transformers`, this is handled
                                    automatically using information from `model.config`.
                    threshold(float): threshold for marking a neuron as activated
                    coarse_coverage(bool):  measure neuron coverage at the level of layer outputs
    """

    def __init__(
        self,
        test_model="textattack/bert-base-uncased-ag-news",
        tokenizer=None,
				num_labels = 2,
        max_seq_len=-1,
        threshold=0.0,
        coarse_coverage=True,
    ):

        self.coarse_coverage = coarse_coverage

        config = transformers.AutoConfig.from_pretrained(
            test_model, output_hidden_states=True,num_labels=num_labels
        )
        if config.model_type in COVERAGE_MODEL_TYPES:
            self.test_model = (
                transformers.AutoModelForSequenceClassification.from_pretrained(
                    test_model, config=config
                )
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                test_model, use_fast=True
            )
            self.model_type = self.test_model.config.model_type
            self.max_seq_len = (
                max_seq_len
                if max_seq_len != -1
                else self.test_model.config.max_position_embeddings
            )
        else:
            raise ValueError(
                "`neuronCoverage` only accepts models in "
                + ",".join(COVERAGE_MODEL_TYPES)
            )

        self.test_model.to(textattack.shared.utils.device)
        self.threshold = threshold
        self.test_model.eval()
        self.coverage_tracker = self._init_coverage()

    def _init_coverage(self):
        """Initialize `coverage_tracker` dictionary

        Returns:
        `coverage_tracker`(dict): a dictionary with key: neuron and value: (bool) intialized False
        """
        coverage_tracker = defaultdict(bool)

        for bert_layer_index in range(self.test_model.config.num_hidden_layers):
            coverage_tracker[(bert_layer_index, "output")] = torch.zeros(
                (self.max_seq_len, self.test_model.config.hidden_size), dtype=bool
            ).to(textattack.shared.utils.device)
        coverage_tracker["classifier"] = torch.zeros(
            (len(self.test_model.config.label2id)), dtype=bool
        ).to(textattack.shared.utils.device)
        coverage_tracker["embedding"] = torch.zeros(
            (self.max_seq_len, self.test_model.config.hidden_size), dtype=bool
        ).to(textattack.shared.utils.device)

        return coverage_tracker

    def _eval(self, text):
        """Update `coverage_tracker` for input `text` for coarse coverage
        Args:
                `text`(str): text to update neuron coverage of.

        """
        encodings = self.tokenizer(text, return_tensors="pt")
        if self.max_seq_len > 0:
            input_ids = encodings.input_ids[:, : self.max_seq_len]
            attention_mask = encodings.attention_mask[:, : self.max_seq_len]

        input_ids = input_ids.to(textattack.shared.utils.device)
        attention_mask = attention_mask.to(textattack.shared.utils.device)
        outputs = self.test_model(input_ids, attention_mask=attention_mask)
        return outputs

    def _update_coarse_coverage(self, text):
        """Update `coverage_tracker` for input `text` for coarse coverage
        Args:
                `text`(str): text to update neuron coverage of.

        """
        encodings = self.tokenizer(text, return_tensors="pt")
        if self.max_seq_len > 0:
            input_ids = encodings.input_ids[:, : self.max_seq_len]
            attention_mask = encodings.attention_mask[:, : self.max_seq_len]

        input_ids = input_ids.to(textattack.shared.utils.device)
        attention_mask = attention_mask.to(textattack.shared.utils.device)
        outputs = self.test_model(input_ids, attention_mask=attention_mask)
        sentence_length = outputs[1][0][0, ...].size(0)

        def scale(layer_outputs, rmax=1, rmin=0):
            divider = layer_outputs.max() - layer_outputs.min()

            if divider == 0:
                return torch.zeros_like(layer_outputs)

            X_std = (layer_outputs - layer_outputs.min()) / divider

            X_scaled = X_std * (rmax - rmin) + rmin
            return X_scaled

        self.coverage_tracker[("embedding")][0:sentence_length, ...] = torch.where(
            scale(outputs[1][0][0, ...]) > self.threshold,
            torch.ones(
                (sentence_length, self.test_model.config.hidden_size), dtype=bool
            ).to(textattack.shared.utils.device),
            self.coverage_tracker[("embedding")][0:sentence_length, ...],
        )
        for h_index, hidden_vector in enumerate(outputs[1][1:]):

            self.coverage_tracker[(h_index, "output")][
                0:sentence_length, ...
            ] = torch.where(
                scale(hidden_vector[0, ...]) > self.threshold,
                torch.ones(
                    (sentence_length, self.test_model.config.hidden_size), dtype=bool
                ).to(textattack.shared.utils.device),
                self.coverage_tracker[(h_index, "output")][0:sentence_length, ...],
            )

        self.coverage_tracker["classifier"] = torch.where(
            scale(outputs[0][0, ...]) > self.threshold,
            torch.ones((len(self.test_model.config.label2id)), dtype=bool).to(
                textattack.shared.utils.device
            ),
            self.coverage_tracker["classifier"],
        )

    def _update_refined_coverage(self, text):
        """Update `coverage_tracker` for input `text` for refined coverage
        Args:
                `text`(str): text to update neuron coverage of.

        """

    def _compute_coverage(self):
        """Calculate `neuron_coverage` for current model"""

        neuron_coverage = sum(
            [entry.sum().item() for entry in self.coverage_tracker.values()]
        ) / sum([entry.numel() for entry in self.coverage_tracker.values()])

        return neuron_coverage

    def _update_coverage(self, text):
        """Update `coverage_tracker` for input `text`
        Args:
                `text`(str): text to update neuron coverage of.

        """
        if self.coarse_coverage:
            self._update_coarse_coverage(text)
        else:
            pass

    def __call__(self, testset):
        """
        Returns neuron of `testset`
        Args:
                        testset: Iterable of strings
        Returns:
                        neuron coverage (float)
        """
        for t in tqdm(testset):

            self._update_coverage(t[0]["text"])
        neuron_coverage = self._compute_coverage()
        return neuron_coverage
