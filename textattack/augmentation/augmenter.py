"""
Augmenter Class
===================

"""
from copy import deepcopy
import random
import sys
from typing import List

import tqdm

from textattack.constraints import PreTransformationConstraint
from textattack.shared import AttackedText, utils


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
    """

    def __init__(
        self,
        transformation,
        constraints=[],
        pct_words_to_swap=0.1,
        transformations_per_example=1,
        high_yield=False,
    ):
        assert (
            transformations_per_example > 0
        ), "transformations_per_example must be a positive integer"
        assert 0.0 <= pct_words_to_swap <= 1.0, "pct_words_to_swap must be in [0., 1.]"
        self.transformation = transformation
        self.pct_words_to_swap = pct_words_to_swap
        self.transformations_per_example = transformations_per_example
        self.high_yield = high_yield

        self.constraints = []
        self.pre_transformation_constraints = []
        for constraint in constraints:
            if isinstance(constraint, PreTransformationConstraint):
                self.pre_transformation_constraints.append(constraint)
            else:
                self.constraints.append(constraint)

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
        for _ in range(self.transformations_per_example):
            current_text = attacked_text
            words_swapped = len(current_text.attack_attrs["modified_indices"])

            while words_swapped < num_words_to_swap:
                transformed_texts = self.transformation(
                    current_text, self.pre_transformation_constraints
                )

                # Get rid of transformations we already have
                transformed_texts = [
                    t for t in transformed_texts if t not in all_transformed_texts
                ]

                # Filter out transformations that don't match the constraints.
                transformed_texts = self._filter_transformations(
                    transformed_texts, current_text, original_text
                )

                # if there's no more transformed texts after filter, terminate
                if not len(transformed_texts):
                    break

                current_text = random.choice(transformed_texts)

                # update words_swapped based on modified indices
                words_swapped = max(
                    len(current_text.attack_attrs["modified_indices"]),
                    words_swapped + 1,
                )
            all_transformed_texts.add(current_text)
        return sorted([at.printable_text() for at in all_transformed_texts])

    def augment_many(self, text_list: List[str]) -> List[List[str]]:
        """Returns all possible augmentations of a list of strings according to
        ``self.transformation``.

        Args:
            text_list (list(string)): a list of strings for data augmentation

        Returns a list(string) of augmented texts.
        """
        attacked_texts = [AttackedText(text) for text in text_list]
        dict_of_all_transformed_texts = {text: set() for text in text_list}
        list_of_num_words_to_swap = [
            max(int(self.pct_words_to_swap * len(attacked_text.words)), 1)
            for attacked_text in attacked_texts
        ]
        for _ in range(self.transformations_per_example):
            current_texts = deepcopy(attacked_texts)
            current_attacked_texts = deepcopy(attacked_texts)
            list_of_words_swapped = [
                len(current_text.attack_attrs["modified_indices"])
                for current_text in current_texts
            ]
            dict_transformed_texts = dict()
            while list_of_words_swapped:
                list_of_transformed_texts = self.transformation.transform_many(
                    current_texts, self.pre_transformation_constraints
                )
                for index in range(len(current_attacked_texts)):
                    # Get rid of transformations we already have
                    list_of_transformed_texts[index] = [
                        t
                        for t in list_of_transformed_texts[index]
                        if t
                        not in dict_of_all_transformed_texts[
                            current_attacked_texts[index].text
                        ]
                    ]
                    # Filter out transformations that don't match the constraints.
                    list_of_transformed_texts[index] = self._filter_transformations(
                        list_of_transformed_texts[index],
                        current_texts[index],
                        current_attacked_texts[index],
                    )

                    dict_transformed_texts[
                        current_attacked_texts[index].text
                    ] = list_of_transformed_texts[index]

                    # if there's no more transformed texts after filter, terminate
                    if not len(list_of_transformed_texts[index]):
                        list_of_words_swapped[index] = sys.maxsize
                        continue

                    current_texts[index] = random.choice(
                        list_of_transformed_texts[index]
                    )
                    # update words_swapped based on modified indices
                    list_of_words_swapped[index] = max(
                        len(current_texts[index].attack_attrs["modified_indices"]), 1
                    )

                # get all indices that still needs words swapped
                indices_to_swap = [
                    index
                    for index in range(len(current_texts))
                    if list_of_words_swapped[index] < list_of_num_words_to_swap[index]
                ]
                current_attacked_texts = [
                    current_attacked_texts[i] for i in indices_to_swap
                ]
                current_texts = [current_texts[i] for i in indices_to_swap]
                list_of_words_swapped = [
                    list_of_words_swapped[i] for i in indices_to_swap
                ]

            if self.high_yield:
                dict_of_all_transformed_texts = {
                    text: all_transformed_texts | set(dict_transformed_texts[text])
                    if text in dict_transformed_texts.keys()
                    else list()
                    for text, all_transformed_texts in dict_of_all_transformed_texts.items()
                }
            else:
                dict_of_all_transformed_texts = {
                    text: all_transformed_texts
                    | {random.choice(dict_transformed_texts[text])}
                    if dict_transformed_texts[text]
                    else all_transformed_texts
                    for text, all_transformed_texts in dict_of_all_transformed_texts.items()
                }
        return [
            sorted([at.printable_text() for at in transformed_texts])
            for transformed_texts in dict_of_all_transformed_texts.values()
        ]

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
            all_text_list.extend([text] + augmented_texts)
            all_id_list.extend([_id] * (1 + len(augmented_texts)))
        return all_text_list, all_id_list

    def __repr__(self):
        main_str = "Augmenter" + "("
        lines = list()
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
