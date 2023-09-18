from collections import defaultdict
import copy
import itertools
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers

import textattack

from .coverage import ExtrinsicCoverage

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


COVERAGE_MODEL_TYPES = ["bert", "albert", "distilbert", "roberta"]


class neuronMultiSectionCoverage(ExtrinsicCoverage):
    """
    ``neuronMultiSectionCoverage`` measures the neuron coverage acheived by a testset
    Args:
                    test_model(Union[str, torch.nn.Module]): name of the pretrained language model from `transformers`
                                    or the actual test model as a `torch.nn.Module` class. Default is "bert base uncased" from `transformers`.
                    tokenizer (:obj:``, optional): If `test_model` is not a pretrained model from `transformers, need to provide
                                    the tokenizer here.
                    max_seq_len (int): Maximum sequence length accepted by the model to be tested.	However, if you are using a pretrained model from `transformers`, this is handled
                                    automatically using information from `model.config`.
                    threshold(float): threshold for marking a neuron as activated
                    coverage(str):	measure type of neuron coverage at the level of layer outputs
    """

    def __init__(
        self,
        test_model="textattack/bert-base-uncased-ag-news",
        tokenizer=None,
        max_seq_len=-1,
        threshold=0.0,
        num_labels=2,
        coverage="multisection",
        pre_limits=False,
        bins_attention=4,
        bins_word=4,
        min_value=np.inf,
        max_value=-np.inf,
        bz=128,
        word_mask=False,
    ):
        self.coverage = coverage

        self.word_mask = word_mask
        self.pre_limits = pre_limits
        self.bins_attention = bins_attention
        self.bins_word = bins_word  # number of sections for each neuron
        self.max_seq_len = 128
        self.model_type = "bert"

        config = transformers.AutoConfig.from_pretrained(
            test_model, output_hidden_states=True, num_labels=num_labels
        )
        if config.model_type in COVERAGE_MODEL_TYPES:
            self.test_model = (
                transformers.AutoModelForSequenceClassification.from_pretrained(
                    test_model, config=config
                )
            )
            self.test_model.tokenizer = transformers.AutoTokenizer.from_pretrained(
                test_model
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

        # initialize min and max for coverage
        min_attention_value = min_value
        max_attention_value = max_value
        if pre_limits:
            min_attention_value = 0.0
            max_attention_value = 1.0

        self.coverage_word_dicts = torch.zeros(
            (self.bins_word + 3, 13, self.max_seq_len, 768)
        )
        self.coverage_attention_dicts = torch.zeros(
            (self.bins_attention + 3, 12, 12, self.max_seq_len, self.max_seq_len)
        )
        self.min_word_coverage_tracker = torch.zeros((13, self.max_seq_len, 768)).fill_(
            min_value
        )
        self.min_attention_coverage_tracker = torch.zeros(
            (12, 12, self.max_seq_len, self.max_seq_len)
        ).fill_(min_attention_value)

        self.max_word_coverage_tracker = torch.zeros((13, self.max_seq_len, 768)).fill_(
            max_value
        )
        self.max_attention_coverage_tracker = torch.zeros(
            (12, 12, self.max_seq_len, self.max_seq_len)
        ).fill_(max_attention_value)

        if "snac" in self.coverage:
            self.k_m = 2
        if "nbc" in self.coverage:
            self.k_m = 1
        """
				for i in range(self.bins_word):
						word_tracker = self._init_word_coverage(fill_value=0.0)
						self.coverage_word_dicts.append(word_tracker)
				for i in range(self.bins_attention):
						attention_tracker = self._init_attention_coverage(fill_value=0.0)
						self.coverage_attention_dicts.append(attention_tracker)
				"""

    def _init_word_coverage(self, fill_value):
        """Initialize `coverage_tracker` dictionary.

        Returns:
                        `coverage_tracker`(dict): a dictionary with key: neuron and value: (bool) intialized False
        """
        coverage_word_tracker = torch.zeros_like(self.coverage_word_dicts)

        """
				coverage_tracker["classifier"] = (
							torch.zeros((len(self.test_model.config.label2id)), requires_grad=False)
							.fill_(fill_value)
							.to(textattack.shared.utils.device)
							.detach()
					)
				"""
        # embedding is L X H

        """
					coverage_tracker["classifier"] = (
							torch.zeros((len(self.test_model.config.label2id)), requires_grad=False)
							.fill_(fill_value)
							.to(textattack.shared.utils.device)
							.detach()
				)
				"""

        return coverage_word_tracker

    def _init_attention_coverage(self, fill_value):
        """Initialize `coverage_tracker` dictionary.

        Returns:
                        `coverage_tracker`(dict): a dictionary with key: neuron and value: (bool) intialized False
        """
        # attention neurons
        coverage_attention_tracker = torch.zeros_like(self.coverage_attention_dicts)
        return coverage_attention_tracker

    def _update_initial_word_coverage(
        self, embeddings, word_mask=None, interaction_mask=None
    ):
        """Update `coverage_tracker` for input `text` for coarse coverage
                        Args:
        `text`(str): text to update neuron coverage of.

        """

        """
				encodings = self.test_model.tokenizer(text, return_tensors="pt")
				if self.max_seq_len > 0:
						input_ids = encodings.input_ids[:, : self.max_seq_len]
						attention_mask = encodings.attention_mask[:, : self.max_seq_len]

				input_ids = input_ids.to(textattack.shared.utils.device)
				attention_mask = attention_mask.to(textattack.shared.utils.device)
				outputs = self.test_model(input_ids, attention_mask=attention_mask)
				outputs[1][0]
				"""

        sentence_length = embeddings[0][0, ...].size(0)

        embeddings = [e.unsqueeze(1) for e in embeddings]

        embeddings = torch.cat(embeddings, dim=1).cpu()

        if self.word_mask:
            indices_to_fill = [int(index) for index in range(sentence_length)]
        else:
            indices_to_fill = [index for index in range(sentence_length)]
        # print(embeddings,, self.max_word_coverage_tracker.device)
        self.max_word_coverage_tracker[:, indices_to_fill, :] = torch.where(
            torch.max(embeddings, dim=0).values.detach()
            > self.max_word_coverage_tracker[:, indices_to_fill, :],
            torch.max(embeddings, dim=0).values.detach(),
            self.max_word_coverage_tracker[:, indices_to_fill, :],
        )
        self.min_word_coverage_tracker[:, indices_to_fill, :] = torch.where(
            torch.min(embeddings, dim=0).values.detach()
            < self.min_word_coverage_tracker[:, indices_to_fill, :],
            torch.min(embeddings, dim=0).values.detach(),
            self.min_word_coverage_tracker[:, indices_to_fill, :],
        )

        """
				self.max_coverage_tracker["classifier"] = torch.where(
						(outputs[0][0, ...].detach()) > self.max_coverage_tracker["classifier"],
						outputs[0][0, ...].detach(),
						self.max_coverage_tracker["classifier"],
				)
				"""

    def _update_initial_attention_coverage(self, all_attentions):
        """Update `coverage_tracker` for input `text` for coarse coverage
                        Args:
        `text`(str): text to update neuron coverage of.

        """

        # all_attentions	= list of attentions of size B X H X L X L

        sentence_length = all_attentions[0][0, 0, ...].size(-1)
        all_attentions = torch.cat(
            [a.unsqueeze(1) for a in all_attentions], dim=1
        )  # B X LA X HD X L X L
        all_attentions_max = torch.max(all_attentions, dim=0).values.cpu()
        all_attentions_min = torch.min(all_attentions, dim=0).values.cpu()
        self.max_attention_coverage_tracker = torch.where(
            all_attentions_max > self.max_attention_coverage_tracker,
            all_attentions_max,
            self.max_attention_coverage_tracker,
        )
        self.min_attention_coverage_tracker = torch.where(
            all_attentions_min < self.min_attention_coverage_tracker,
            all_attentions_min,
            self.min_attention_coverage_tracker,
        )

    def _update_initial_coverage(
        self, all_hidden_states, all_attentions, word_mask=None
    ):
        """Update `coverage_tracker` for input `text`
                        Args:
        `text`(str): text to update neuron coverage of.

        """

        self._update_initial_word_coverage(all_hidden_states, word_mask)

        self._update_initial_attention_coverage(all_attentions)

    def initialize_from_training_dataset(self, trainset, masks=None, bz=1):
        """Update coverage from training dataset
        `trainset`(list[str]): training dataset coverage statistics


        """
        mask_no = 0

        start = 0
        with torch.no_grad():
            for t in tqdm(trainset):
                if mask_no + bz >= len(trainset):
                    end = len(trainset)
                else:
                    end = start + bz
                if start >= end or start > len(trainset):
                    break
                # print('current indices : ', trainset[start:end], start, end, len(trainset))
                encodings = self.test_model.tokenizer(
                    trainset[start:end],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_seq_len,
                )

                if self.max_seq_len > 0:
                    input_ids = encodings.input_ids[:, : self.max_seq_len]
                    attention_mask = encodings.attention_mask[:, : self.max_seq_len]

                input_ids = input_ids.to(textattack.shared.utils.device)
                attention_mask = attention_mask.to(textattack.shared.utils.device)

                outputs = self.test_model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    output_hidden_states=True,
                )
                all_hidden_states, all_attentions = outputs[-2:]
                self._update_initial_coverage(
                    all_hidden_states, all_attentions, masks[start:end]
                )
                start = end

        self.training_word_coverage_dicts = copy.deepcopy(self.coverage_word_dicts)
        self.training_attention_coverage_dicts = copy.deepcopy(
            self.coverage_attention_dicts
        )

    def _eval(self, text):
        """Update `coverage_tracker` for input `text` for coarse coverage
                        Args:
        `text`(str): text to update neuron coverage of.

        """
        encodings = self.test_model.tokenizer(text, return_tensors="pt")
        if self.max_seq_len > 0:
            input_ids = encodings.input_ids[:, : self.max_seq_len]
            attention_mask = encodings.attention_mask[:, : self.max_seq_len]

        input_ids = input_ids.to(textattack.shared.utils.device)
        attention_mask = attention_mask.to(textattack.shared.utils.device)
        outputs = self.test_model(input_ids, attention_mask=attention_mask)
        return outputs

    def _update_word_coverage(self, all_hidden_states, word_mask=None):
        """Update `coverage_tracker` for input `text` for coarse coverage
                        Args:
        `text`(str): text to update neuron coverage of.



        a = time.time()
        encodings = self.test_model.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length = self.max_seq_len)
        if self.max_seq_len > 0:
                        input_ids = encodings.input_ids[:, : self.max_seq_len]
                        attention_mask = encodings.attention_mask[:, : self.max_seq_len]

        input_ids = input_ids.to(textattack.shared.utils.device)
        attention_mask = attention_mask.to(textattack.shared.utils.device)
        outputs = self.test_model(input_ids, attention_mask=attention_mask)
        b = time.time()

        sentence_length = outputs[1][0][0, ...].size(0)
        """
        hidden_vectors = torch.cat([o.unsqueeze(1) for o in all_hidden_states], dim=1)
        sentence_length = hidden_vectors.size(2)
        # print('size of output hidden bectors: ', hidden_vectors.size())
        if self.word_mask:
            indices_to_fill = [index for index in range(sentence_length)]
        else:
            indices_to_fill = [index for index in range(sentence_length)]
        current_coverage_tracker = self._init_word_coverage(fill_value=0)
        a = time.time()
        section_length = (
            self.max_word_coverage_tracker[:, indices_to_fill, :]
            - self.min_word_coverage_tracker[:, indices_to_fill, :]
        ) / self.bins_word
        section_length = section_length.unsqueeze(0).repeat(
            hidden_vectors.size(0), 1, 1, 1
        )
        # print('section length: ', section_length.size())
        section_index = torch.where(
            section_length > 0,
            (
                torch.floor(
                    (
                        hidden_vectors.cpu().detach()
                        - self.min_word_coverage_tracker[:, indices_to_fill, :]
                    )
                    / section_length
                )
            ),
            torch.zeros_like(hidden_vectors.cpu().detach(), requires_grad=False) - 1,
        ).long()
        # print('section index: ', section_index.size())

        # section_index = torch.where(section_index, section_index, self.bins_word + 1)
        # section_index = torch.where(section_index>0, section_index, torch.zeros_like(section_index) + self.bins_word + 1)
        section_index = torch.where(
            section_index < self.bins_word,
            section_index,
            torch.zeros_like(section_index) + self.bins_word + 2,
        )
        section_index = torch.where(
            section_index > 0,
            section_index,
            torch.zeros_like(section_index) + self.bins_word + 1,
        )

        # print('section index: ', section_index.size())

        temp_store_activations = torch.max(
            (F.one_hot(section_index, num_classes=self.bins_word + 3)).permute(
                0, 4, 1, 2, 3
            ),
            dim=0,
        ).values

        # print('Temp Store Activations: ', temp_store_activations.size())
        self.coverage_word_dicts += temp_store_activations
        del temp_store_activations
        del current_coverage_tracker

    def _update_attention_coverage(self, all_attentions, masks):
        """Update `coverage_tracker` for input `text` for coarse coverage
                        Args:
        `text`(str): text to update neuron coverage of.


        encodings = self.test_model.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length = self.max_seq_len)
        if self.max_seq_len > 0:
                        input_ids = encodings.input_ids[:, : self.max_seq_len]
                        attention_mask = encodings.attention_mask[:, : self.max_seq_len]

        input_ids = input_ids.to(textattack.shared.utils.device)
        attention_mask = attention_mask.to(textattack.shared.utils.device)
        outputs = self.test_model(input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states = True)

        all_hidden_states, all_attentions = outputs[-2:]
        # all_attentions	= list of attentions of size B X H X L X L

        """
        sentence_length = all_attentions[0][0, 0, ...].size(-1)

        all_attentions = torch.cat(
            [a.unsqueeze(1) for a in all_attentions], dim=1
        ).cpu()[:, :, 0:sentence_length, 0:sentence_length]
        # B X layers X heads X l X l
        # print('attentions size: ', all_attentions.size())
        current_coverage_tracker = self._init_attention_coverage(fill_value=0)

        section_length = (
            self.max_attention_coverage_tracker[
                :, :, 0:sentence_length, 0:sentence_length
            ]
            - self.min_attention_coverage_tracker[
                :, :, 0:sentence_length, 0:sentence_length
            ]
        ) / self.bins_attention
        section_length = section_length.unsqueeze(0).repeat(
            all_attentions.size(0), 1, 1, 1, 1
        )
        # print(' section length: ', section_length.size())
        section_index = torch.where(
            section_length > 0,
            (
                torch.floor(
                    (
                        all_attentions.cpu().detach()
                        - self.min_attention_coverage_tracker
                    )
                    / section_length
                )
            ),
            torch.zeros_like(all_attentions.cpu().detach(), requires_grad=False) - 1,
        ).long()

        # print('section index: ', section_index.size())
        section_index = torch.where(
            section_index < self.bins_attention,
            section_index,
            torch.zeros_like(section_index) + self.bins_attention + 2,
        )
        section_index = torch.where(
            section_index > 0,
            section_index,
            torch.zeros_like(section_index) + self.bins_word + 1,
        )
        temp_storage_activations = torch.max(
            (F.one_hot(section_index, num_classes=self.bins_attention + 3)).permute(
                0, 5, 1, 2, 3, 4
            ),
            dim=0,
        ).values
        # print(' temp storage activations: ', temp_storage_activations.size())
        self.coverage_attention_dicts += temp_storage_activations
        del temp_storage_activations
        del current_coverage_tracker

    def _compute_coverage(self):
        """Calculate `neuron_coverage` for current model."""
        neuron_word_coverage, neuron_word_coverage_total = 0.0, 0.0
        neuron_attention_coverage, neuron_attention_coverage_total = 0.0, 0.0
        neuron_word_coverage += np.count_nonzero(self.coverage_word_dicts.numpy())
        neuron_word_coverage_total += self.coverage_word_dicts.numel()

        neuron_attention_coverage += np.count_nonzero(
            self.coverage_attention_dicts.numpy()
        )
        neuron_attention_coverage_total += self.coverage_attention_dicts.numel()

        neuron_coverage = neuron_word_coverage + neuron_attention_coverage
        # print('Word and Attention Only: ', neuron_word_coverage , neuron_attention_coverage)
        neuron_coverage_total = (
            neuron_word_coverage_total + neuron_attention_coverage_total
        )
        # print('Total Word and Attention Only: ', neuron_word_coverage_total , neuron_attention_coverage_total)
        return neuron_coverage / neuron_coverage_total

    def _compute_vector(self):
        """Calculate `neuron_coverage` for current model."""
        neuron_coverage_vector = []
        for section in self.coverage_word_dicts:
            for entry in section.values():
                neuron_coverage_vector += [
                    entry_val.item() for entry_val in entry.flatten()
                ]
        for section in self.coverage_attention_dicts:
            for entry in section.values():
                neuron_coverage_vector += [
                    entry_val.item() for entry_val in entry.flatten()
                ]

        return neuron_coverage_vector

    def _update_coverage(self, text, word_mask=None):
        """Update `coverage_tracker` for input `text`
                        Args:
        `text`(str): text to update neuron coverage of.

        """

        self._update_word_coverage(text, word_mask)
        self._update_attention_coverage(text)

    def __call__(self, testset, masks=None, bz=1):
        """
        Returns neuron of `testset`
        Args:
                        testset: Iterable of strings
        Returns:
                        neuron coverage (float)
        """
        # # # print('*'*50)
        # # # print('Updating Coverage using test set: ')
        mask_no, start = 0, 0
        with torch.no_grad():
            for t in tqdm(testset):
                if mask_no + bz >= len(testset):
                    end = len(testset)
                else:
                    end = start + bz
                if start >= end or start > len(testset):
                    break

                encodings = self.test_model.tokenizer(
                    testset[start:end],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_seq_len,
                )

                if self.max_seq_len > 0:
                    input_ids = encodings.input_ids[:, : self.max_seq_len]
                    attention_mask = encodings.attention_mask[:, : self.max_seq_len]

                input_ids = input_ids.to(textattack.shared.utils.device)
                attention_mask = attention_mask.to(textattack.shared.utils.device)

                outputs = self.test_model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    output_hidden_states=True,
                )
                all_hidden_states, all_attentions = outputs[-2:]
                self._update_word_coverage(all_hidden_states, masks[start:end])
                self._update_attention_coverage(all_attentions, masks[start:end])

                start = end

        # # # print('*'*50)
        # # # print()
        # # # print('*'*50)
        # # # print('Computing Coverage: ')
        neuron_coverage = self._compute_coverage()
        # # # print('*'*50)
        return neuron_coverage

    def vector(self, testset, start=False):
        """
        Returns neuron of `testset`
        Args:
                        testset: Iterable of strings
        Returns:
                        neuron coverage (float)
        """
        # # # print('*'*50)
        if start:
            self.coverage_word_dicts = copy.deepcopy(self.training_word_coverage_dicts)
            self.coverage_attention_dicts = copy.deepcopy(
                self.training_attention_coverage_dicts
            )
        # # # print('Updating Coverage using test set: ')
        # # # print('#'*100)
        # # # print(len(testset))
        # # # print(testset)
        # # # print('#'*100)
        for t in tqdm(testset):
            # # # print(t)
            self._update_coverage(t)

        # # # print('*'*50)
        # # # print()
        # # # print('*'*50)
        # # # print('Computing Coverage: ')
        neuron_coverage = self._compute_vector()
        # # print('*'*50)
        return neuron_coverage
