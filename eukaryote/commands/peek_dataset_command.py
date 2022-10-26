"""

PeekDatasetCommand class
==============================

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import collections
import re

import numpy as np

import eukaryote
from eukaryote.commands import TextAttackCommand


def _cb(s):
    return eukaryote.shared.utils.color_text(str(s), color="blue", method="ansi")


logger = eukaryote.shared.logger


class PeekDatasetCommand(TextAttackCommand):
    """The peek dataset module:

    Takes a peek into a dataset in textattack.
    """

    def run(self, args):
        UPPERCASE_LETTERS_REGEX = re.compile("[A-Z]")

        dataset_args = eukaryote.DatasetArgs(**vars(args))
        dataset = eukaryote.DatasetArgs._create_dataset_from_args(dataset_args)

        num_words = []
        attacked_texts = []
        data_all_lowercased = True
        outputs = []
        for inputs, output in dataset:
            at = eukaryote.shared.AttackedText(inputs)
            if data_all_lowercased:
                # Test if any of the letters in the string are lowercase.
                if re.search(UPPERCASE_LETTERS_REGEX, at.text):
                    data_all_lowercased = False
            attacked_texts.append(at)
            num_words.append(len(at.words))
            outputs.append(output)

        logger.info(f"Number of samples: {_cb(len(attacked_texts))}")
        logger.info("Number of words per input:")
        num_words = np.array(num_words)
        logger.info(f'\t{("total:").ljust(8)} {_cb(num_words.sum())}')
        mean_words = f"{num_words.mean():.2f}"
        logger.info(f'\t{("mean:").ljust(8)} {_cb(mean_words)}')
        std_words = f"{num_words.std():.2f}"
        logger.info(f'\t{("std:").ljust(8)} {_cb(std_words)}')
        logger.info(f'\t{("min:").ljust(8)} {_cb(num_words.min())}')
        logger.info(f'\t{("max:").ljust(8)} {_cb(num_words.max())}')
        logger.info(f"Dataset lowercased: {_cb(data_all_lowercased)}")

        logger.info("First sample:")
        print(attacked_texts[0].printable_text(), "\n")
        logger.info("Last sample:")
        print(attacked_texts[-1].printable_text(), "\n")

        logger.info(f"Found {len(set(outputs))} distinct outputs.")
        if len(outputs) < 20:
            print(sorted(set(outputs)))

        logger.info("Most common outputs:")
        for i, (key, value) in enumerate(collections.Counter(outputs).most_common(20)):
            print("\t", str(key)[:5].ljust(5), f" ({value})")

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "peek-dataset",
            help="show main statistics about a dataset",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser = eukaryote.DatasetArgs._add_parser_args(parser)
        parser.set_defaults(func=PeekDatasetCommand())
