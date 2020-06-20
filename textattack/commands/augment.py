from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
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
}


class AugmentCommand(TextAttackCommand):
    """
    The TextAttack attack module:
    
        A command line parser to run data augmentation from user specifications.
    """

    def run(self, args):
        """ Reads in a CSV, performs augmentation, and outputs an augmented CSV.
        
            Preserves all columns except for the input (augmneted) column.
        """
        textattack.shared.utils.set_seed(args.random_seed)
        start_time = time.time()
        # Validate input/output paths.
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"Can't find CSV at location {args.csv}")
        if os.path.exists(args.outfile):
            if args.overwrite:
                textattack.shared.logger.info(f"Preparing to overwrite {args.outfile}.")
            else:
                raise OSError(f"Outfile {args.outfile} exists and --overwrite not set.")
        # Read in CSV file as a list of dictionaries. Use the CSV sniffer to
        # try and automatically infer the correct CSV format.
        csv_file = open(args.csv, "r")
        dialect = csv.Sniffer().sniff(csv_file.readline(), delimiters=";,")
        csv_file.seek(0)
        rows = [
            row
            for row in csv.DictReader(csv_file, dialect=dialect, skipinitialspace=True)
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
        # Augment all examples.
        augmenter = eval(AUGMENTATION_RECIPE_NAMES[args.recipe])(
            num_words_to_swap=args.num_words_to_swap,
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
            "--csv", help="input csv file to augment", type=str, required=True
        )
        parser.add_argument(
            "--input-column",
            "--i",
            help="csv input column to be augmented",
            type=str,
            required=True,
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
            "--num-words-to-swap",
            "--n",
            help="words to swap out for each augmented example",
            type=int,
            default=2,
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
            "--random-seed", default=42, type=int, help="random seed to set"
        )
        parser.set_defaults(func=AugmentCommand())
