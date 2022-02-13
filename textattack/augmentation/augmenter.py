"""
Augmenter Class
===================
"""
import random

import tqdm

from textattack.constraints import PreTransformationConstraint
from textattack.metrics.quality_metrics import Perplexity, USEMetric
from textattack.shared import AttackedText, utils
from textattack.transformations import CompositeTransformation, Transformation
import lru
import torch


class Augmenter:
    """A class for performing data augmentation using TextAttack.

    Returns all possible transformations for a given string. Currently only
        supports transformations which are word swaps.

    Args:
        transformation (textattack.Transformation): the transformation
            that suggests new texts from an input.
        constraints: (list(textattack.Constraint)): constraints
            that each transformation must meet
        pct_words_to_swap: (float): [0., 1.], percentage of words to swap per augmented example
        transformations_per_example: (int): Maximum number of augmentations
            per input
        high_yield: Whether to return a set of augmented texts that will be relatively similar, or to return only a
            single one.
        fast_augment: Stops additional transformation runs when number of successful augmentations reaches
            transformations_per_example
        advanced_metrics: return perplexity and USE Score of augmentation

    Example::
        >>> from textattack.transformations import WordSwapRandomCharacterDeletion, WordSwapQWERTY, CompositeTransformation
        >>> from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
        >>> from textattack.augmentation import Augmenter

        >>> transformation = CompositeTransformation([WordSwapRandomCharacterDeletion(), WordSwapQWERTY()])
        >>> constraints = [RepeatModification(), StopwordModification()]

        >>> # initiate augmenter
        >>> augmenter = Augmenter(
        ...     transformation=transformation,
        ...     constraints=constraints,
        ...     pct_words_to_swap=0.5,
        ...     transformations_per_example=3
        ... )

        >>> # additional parameters can be modified if not during initiation
        >>> augmenter.enable_advanced_metrics = True
        >>> augmenter.fast_augment = True
        >>> augmenter.high_yield = True

        >>> s = 'What I cannot create, I do not understand.'
        >>> results = augmenter.augment(s)

        >>> augmentations = results[0]
        >>> perplexity_score = results[1]
        >>> use_score = results[2]
    """

    def __init__(
        self,
        transformation,
        constraints=[],
        pct_words_to_swap=0.1,
        transformations_per_example=1,
        high_yield=False,
        fast_augment=False,
        enable_advanced_metrics=False,
        transformation_cache_size=2 ** 15,
        constraint_cache_size=2 ** 15,
        use_cache=True
    ):
        assert (
            transformations_per_example > 0
        ), "transformations_per_example must be a positive integer"
        assert 0.0 <= pct_words_to_swap <= 1.0, "pct_words_to_swap must be in [0., 1.]"
        self.transformation = transformation
        self.pct_words_to_swap = pct_words_to_swap
        self.transformations_per_example = transformations_per_example
        self.constraints = []
        self.pre_transformation_constraints = []
        self.high_yield = high_yield
        self.fast_augment = fast_augment
        self.advanced_metrics = enable_advanced_metrics
        self.use_cache = use_cache

        for constraint in constraints:
            if isinstance(constraint, PreTransformationConstraint):
                self.pre_transformation_constraints.append(constraint)
            else:
                self.constraints.append(constraint)

        if self.use_cache:
            if not self.transformation.deterministic:
                self.use_transformation_cache = False
            elif isinstance(self.transformation, CompositeTransformation):
                self.use_transformation_cache = True
                for t in self.transformation.transformations:
                    if not t.deterministic:
                        self.use_transformation_cache = False
                        break
            else:
                self.use_transformation_cache = True
        else:
            self.use_transformation_cache = False

        if self.use_transformation_cache:
            self.transformation_cache_size = transformation_cache_size
            self.transformation_cache = lru.LRU(transformation_cache_size)
            self.constraint_cache_size = constraint_cache_size
            self.constraints_cache = lru.LRU(constraint_cache_size)

    def _get_transformations_uncached(self, current_text, original_text=None, **kwargs):
        """Applies ``self.transformation`` to ``text``, then filters the list
        of possible transformations through the applicable constraints.

        Args:
            current_text: The current ``AttackedText`` on which to perform the transformations.
            original_text: The original ``AttackedText`` from which the attack started.
        Returns:
            A filtered list of transformations where each transformation matches the constraints
        """
        transformed_texts = self.transformation(
            current_text,
            pre_transformation_constraints=self.pre_transformation_constraints,
            **kwargs,
        )

        return transformed_texts

    def get_transformations(self, current_text, original_text=None, **kwargs):
        """Applies ``self.transformation`` to ``text``, then filters the list
        of possible transformations through the applicable constraints.

        Args:
            current_text: The current ``AttackedText`` on which to perform the transformations.
            original_text: The original ``AttackedText`` from which the attack started.
        Returns:
            A filtered list of transformations where each transformation matches the constraints
        """
        if not self.transformation:
            raise RuntimeError(
                "Cannot call `get_transformations` without a transformation."
            )
        if self.use_transformation_cache:
            cache_key = tuple([current_text] + sorted(kwargs.items()))
            if utils.hashable(cache_key) and cache_key in self.transformation_cache:
                # promote transformed_text to the top of the LRU cache
                self.transformation_cache[cache_key] = self.transformation_cache[
                    cache_key
                ]
                transformed_texts = list(self.transformation_cache[cache_key])
            else:
                transformed_texts = self._get_transformations_uncached(
                    current_text, original_text, **kwargs
                )
                if utils.hashable(cache_key):
                    self.transformation_cache[cache_key] = tuple(transformed_texts)
        else:
            transformed_texts = self._get_transformations_uncached(
                current_text, original_text, **kwargs
            )

        return self.filter_transformations(
            transformed_texts, current_text, original_text
        )

    def _filter_transformations_uncached(
        self, transformed_texts, current_text, original_text=None
    ):
        """Filters a list of potential transformaed texts based on
        ``self.constraints``

        Args:
            transformed_texts: A list of candidate transformed ``AttackedText`` to filter.
            current_text: The current ``AttackedText`` on which the transformation was applied.
            original_text: The original ``AttackedText`` from which the attack started.
        """
        filtered_texts = transformed_texts[:]
        for C in self.constraints:
            if len(filtered_texts) == 0:
                break
            if C.compare_against_original:
                if not original_text:
                    raise ValueError(
                        f"Missing `original_text` argument when constraint {type(C)} is set to compare against `original_text`"
                    )

                filtered_texts = C.call_many(filtered_texts, original_text)
            else:
                filtered_texts = C.call_many(filtered_texts, current_text)
        # Default to false for all original transformations.
        for original_transformed_text in transformed_texts:
            self.constraints_cache[(current_text, original_transformed_text)] = False
        # Set unfiltered transformations to True in the cache.
        for filtered_text in filtered_texts:
            self.constraints_cache[(current_text, filtered_text)] = True
        return filtered_texts

    def filter_transformations(
        self, transformed_texts, current_text, original_text=None
    ):
        """Filters a list of potential transformed texts based on
        ``self.constraints`` Utilizes an LRU cache to attempt to avoid
        recomputing common transformations.

        Args:
            transformed_texts: A list of candidate transformed ``AttackedText`` to filter.
            current_text: The current ``AttackedText`` on which the transformation was applied.
            original_text: The original ``AttackedText`` from which the attack started.
        """
        # Remove any occurences of current_text in transformed_texts
        transformed_texts = [
            t for t in transformed_texts if t.text != current_text.text
        ]
        # Populate cache with transformed_texts
        uncached_texts = []
        filtered_texts = []
        for transformed_text in transformed_texts:
            if (current_text, transformed_text) not in self.constraints_cache:
                uncached_texts.append(transformed_text)
            else:
                # promote transformed_text to the top of the LRU cache
                self.constraints_cache[
                    (current_text, transformed_text)
                ] = self.constraints_cache[(current_text, transformed_text)]
                if self.constraints_cache[(current_text, transformed_text)]:
                    filtered_texts.append(transformed_text)
        filtered_texts += self._filter_transformations_uncached(
            uncached_texts, current_text, original_text=original_text
        )
        # Sort transformations to ensure order is preserved between runs
        filtered_texts.sort(key=lambda t: t.text)
        return filtered_texts

    def _filter_transformations(self, transformed_texts, current_text, original_text):
        """Filters a list of ``AttackedText`` objects to include only the ones
        that pass ``self.constraints``."""
        for C in self.constraints:
            if len(transformed_texts) == 0:
                break
            if C.compare_against_original:
                if not original_text:
                    raise ValueError(
                        f"Missing `original_text` argument when constraint {type(C)} is set to compare against "
                        f"`original_text` "
                    )

                transformed_texts = C.call_many(transformed_texts, original_text)
            else:
                transformed_texts = C.call_many(transformed_texts, current_text)
        return transformed_texts

    def augment(self, text):
        """Returns all possible augmentations of ``text`` according to
        ``self.transformation``."""
        attacked_text = AttackedText(text)
        original_text = attacked_text
        all_transformed_texts = set()
        num_words_to_swap = max(
            int(self.pct_words_to_swap * len(attacked_text.words)), 1
        )
        augmentation_results = []
        for _ in range(self.transformations_per_example):
            current_text = attacked_text
            words_swapped = len(current_text.attack_attrs["modified_indices"])

            while words_swapped < num_words_to_swap:
                if self.use_transformation_cache:
                    transformed_texts = self.get_transformations(current_text)
                else:
                    transformed_texts = self.transformation(
                        current_text, self.pre_transformation_constraints
                    )

                # Get rid of transformations we already have
                transformed_texts = [
                    t for t in transformed_texts if t not in all_transformed_texts
                ]

                # Filter out transformations that don't match the constraints.
                if self.use_transformation_cache:
                    transformed_texts = self.filter_transformations(
                        transformed_texts, current_text, original_text
                    )
                else:
                    transformed_texts = self._filter_transformations(
                        transformed_texts, current_text, original_text
                    )

                # if there's no more transformed texts after filter, terminate
                if not len(transformed_texts):
                    break

                # look for all transformed_texts that has enough words swapped
                if self.high_yield or self.fast_augment:
                    ready_texts = [
                        text
                        for text in transformed_texts
                        if len(text.attack_attrs["modified_indices"])
                        >= num_words_to_swap
                    ]
                    for text in ready_texts:
                        all_transformed_texts.add(text)
                    unfinished_texts = [
                        text for text in transformed_texts if text not in ready_texts
                    ]

                    if len(unfinished_texts):
                        current_text = random.choice(unfinished_texts)
                    else:
                        # no need for further augmentations if all of transformed_texts meet `num_words_to_swap`
                        break
                else:
                    current_text = random.choice(transformed_texts)

                # update words_swapped based on modified indices
                words_swapped = max(
                    len(current_text.attack_attrs["modified_indices"]),
                    words_swapped + 1,
                )

            all_transformed_texts.add(current_text)

            # when with fast_augment, terminate early if there're enough successful augmentations
            if (
                self.fast_augment
                and len(all_transformed_texts) >= self.transformations_per_example
            ):
                if not self.high_yield:
                    all_transformed_texts = random.sample(
                        all_transformed_texts, self.transformations_per_example
                    )
                break

        perturbed_texts = sorted([at.printable_text() for at in all_transformed_texts])

        if self.advanced_metrics:
            for transformed_texts in all_transformed_texts:
                augmentation_results.append(
                    AugmentationResult(original_text, transformed_texts)
                )
            perplexity_stats = Perplexity().calculate(augmentation_results)
            use_stats = USEMetric().calculate(augmentation_results)
            return perturbed_texts, perplexity_stats, use_stats

        return perturbed_texts

    def augment_many(self, text_list, show_progress=False):
        """Returns all possible augmentations of a list of strings according to
        ``self.transformation``.

        Args:
            text_list (list(string)): a list of strings for data augmentation
        Returns a list(string) of augmented texts.
        :param show_progress: show process during augmentation
        """
        if show_progress:
            text_list = tqdm.tqdm(text_list, desc="Augmenting data...")
        return [self.augment(text) for text in text_list]

    def augment_text_with_ids(self, text_list, id_list, show_progress=True):
        """Supplements a list of text with more text data.

        Returns the augmented text along with the corresponding IDs for
        each augmented example.
        """
        if len(text_list) != len(id_list):
            raise ValueError("List of text must be same length as list of IDs")
        if self.transformations_per_example == 0:
            return text_list, id_list
        all_text_list = []
        all_id_list = []
        if show_progress:
            text_list = tqdm.tqdm(text_list, desc="Augmenting data...")
        for text, _id in zip(text_list, id_list):
            all_text_list.append(text)
            all_id_list.append(_id)
            augmented_texts = self.augment(text)
            all_text_list.extend
            all_text_list.extend([text] + augmented_texts)
            all_id_list.extend([_id] * (1 + len(augmented_texts)))
        return all_text_list, all_id_list

    def __repr__(self):
        main_str = "Augmenter" + "("
        lines = []
        # self.transformation
        lines.append(utils.add_indent(f"(transformation):  {self.transformation}", 2))
        # self.constraints
        constraints_lines = []
        constraints = self.constraints + self.pre_transformation_constraints
        if len(constraints):
            for i, constraint in enumerate(constraints):
                constraints_lines.append(utils.add_indent(f"({i}): {constraint}", 2))
            constraints_str = utils.add_indent("\n" + "\n".join(constraints_lines), 2)
        else:
            constraints_str = "None"
        lines.append(utils.add_indent(f"(constraints): {constraints_str}", 2))
        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str


class AugmentationResult:
    def __init__(self, text1, text2):
        self.original_result = self.tempResult(text1)
        self.perturbed_result = self.tempResult(text2)

    class tempResult:
        def __init__(self, text):
            self.attacked_text = text
