"""

TextAttack Augment Command
===========================

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentError, ArgumentParser
import csv
import os
import time

import tqdm

import textattack
from textattack.commands import TextAttackCommand

AUGMENTATION_RECIPE_NAMES = {
    "wordnet": "textattack.augmentation.WordNetAugmenter",
    "embedding": "textattack.augmentation.EmbeddingAugmenter",
    "charswap": "textattack.augmentation.CharSwapAugmenter",
    "eda": "textattack.augmentation.EasyDataAugmenter",
    "checklist": "textattack.augmentation.CheckListAugmenter",
    "clare": "textattack.augmentation.CLAREAugmenter",
}


class AugmentCommand(TextAttackCommand):
    """The TextAttack Augment Command module:

    A command line parser to run data augmentation from user
    specifications.
    """

    def run(self, args):
        """Reads in a CSV, performs augmentation, and outputs an augmented CSV.

        Preserves all columns except for the input (augmneted) column.
        """
        if args.interactive:

            print("\nRunning in interactive mode...\n")
            augmenter = eval(AUGMENTATION_RECIPE_NAMES[args.recipe])(
                pct_words_to_swap=args.pct_words_to_swap,
                transformations_per_example=args.transformations_per_example,
            )
            print("--------------------------------------------------------")

            while True:
                print(
                    '\nEnter a sentence to augment, "q" to quit, "c" to view/change arguments:\n'
                )
                text = input()

                if text == "q":
                    break

                elif text == "c":
                    print(
                        f"\nCurrent Arguments:\n\n\t augmentation recipe: {args.recipe}, "
                        f"\n\t pct_words_to_swap: {args.pct_words_to_swap}, "
                        f"\n\t transformations_per_example: {args.transformations_per_example}\n"
                    )

                    change = input(
                        "Enter 'c' again to change arguments, any other keys to opt out\n"
                    )
                    if change == "c":
                        print("\nChanging augmenter arguments...\n")
                        recipe = input(
                            "\tAugmentation recipe name ('r' to see available recipes):  "
                        )
                        if recipe == "r":
                            print("\n\twordnet, embedding, charswap, eda, checklist\n")
                            args.recipe = input("\tAugmentation recipe name:  ")
                        else:
                            args.recipe = recipe

                        args.pct_words_to_swap = float(
                            input("\tPercentage of words to swap (0.0 ~ 1.0):  ")
                        )
                        args.transformations_per_example = int(
                            input("\tTransformations per input example:  ")
                        )

                        print("\nGenerating new augmenter...\n")
                        augmenter = eval(AUGMENTATION_RECIPE_NAMES[args.recipe])(
                            pct_words_to_swap=args.pct_words_to_swap,
                            transformations_per_example=args.transformations_per_example,
                        )
                        print(
                            "--------------------------------------------------------"
                        )

                    continue

                elif not text:
                    continue

                print("\nAugmenting...\n")
                print("--------------------------------------------------------")

                for augmentation in augmenter.augment(text):
                    print(augmentation, "\n")
                print("--------------------------------------------------------")
        else:
            textattack.shared.utils.set_seed(args.random_seed)
            start_time = time.time()
            if not (args.csv and args.input_column):
                raise ArgumentError(
                    "The following arguments are required: --csv, --input-column/--i"
                )
            # Validate input/output paths.
            if not os.path.exists(args.csv):
                raise FileNotFoundError(f"Can't find CSV at location {args.csv}")
            if os.path.exists(args.outfile):
                if args.overwrite:
                    textattack.shared.logger.info(
                        f"Preparing to overwrite {args.outfile}."
                    )
                else:
                    raise OSError(
                        f"Outfile {args.outfile} exists and --overwrite not set."
                    )
            # Read in CSV file as a list of dictionaries. Use the CSV sniffer to
            # try and automatically infer the correct CSV format.
            csv_file = open(args.csv, "r")
            dialect = csv.Sniffer().sniff(csv_file.readline(), delimiters=";,")
            csv_file.seek(0)
            rows = [
                row
                for row in csv.DictReader(
                    csv_file, dialect=dialect, skipinitialspace=True
                )
            ]
            # Validate input column.
            row_keys = set(rows[0].keys())
            if args.input_column not in row_keys:
                raise ValueError(
                    f"Could not find input column {args.input_column} in CSV. Found keys: {row_keys}"
                )
            textattack.shared.logger.info(
                f"Read {len(rows)} rows from {args.csv}. Found columns {row_keys}."
            )

            augmenter = eval(AUGMENTATION_RECIPE_NAMES[args.recipe])(
                pct_words_to_swap=args.pct_words_to_swap,
                transformations_per_example=args.transformations_per_example,
            )

            output_rows = []
            for row in tqdm.tqdm(rows, desc="Augmenting rows"):
                text_input = row[args.input_column]
                if not args.exclude_original:
                    output_rows.append(row)
                for augmentation in augmenter.augment(text_input):
                    augmented_row = row.copy()
                    augmented_row[args.input_column] = augmentation
                    output_rows.append(augmented_row)
            # Print to file.
            with open(args.outfile, "w") as outfile:
                csv_writer = csv.writer(
                    outfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                # Write header.
                csv_writer.writerow(output_rows[0].keys())
                # Write rows.
                for row in output_rows:
                    csv_writer.writerow(row.values())
            textattack.shared.logger.info(
                f"Wrote {len(output_rows)} augmentations to {args.outfile} in {time.time() - start_time}s."
            )

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "augment",
            help="augment text data",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--csv",
            help="input csv file to augment",
            type=str,
            required=False,
            default=None,
        )
        parser.add_argument(
            "--input-column",
            "--i",
            help="csv input column to be augmented",
            type=str,
            required=False,
            default=None,
        )
        parser.add_argument(
            "--recipe",
            "--r",
            help="recipe for augmentation",
            type=str,
            default="embedding",
            choices=AUGMENTATION_RECIPE_NAMES.keys(),
        )
        parser.add_argument(
            "--pct-words-to-swap",
            "--p",
            help="Percentage of words to modify when generating each augmented example.",
            type=float,
            default=0.1,
        )

        parser.add_argument(
            "--transformations-per-example",
            "--t",
            help="number of augmentations to return for each input",
            type=int,
            default=2,
        )
        parser.add_argument(
            "--outfile", "--o", help="path to outfile", type=str, default="augment.csv"
        )
        parser.add_argument(
            "--exclude-original",
            default=False,
            action="store_true",
            help="exclude original example from augmented CSV",
        )
        parser.add_argument(
            "--overwrite",
            default=False,
            action="store_true",
            help="overwrite output file, if it exists",
        )
        parser.add_argument(
            "--interactive",
            default=False,
            action="store_true",
            help="Whether to run attacks interactively.",
        )
        parser.add_argument(
            "--random-seed", default=42, type=int, help="random seed to set"
        )
        parser.set_defaults(func=AugmentCommand())
