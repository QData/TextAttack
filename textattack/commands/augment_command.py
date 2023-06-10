"""

AugmentCommand class
===========================

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentError, ArgumentParser
import csv
import os
import time

import tqdm

import textattack
from textattack.augment_args import AUGMENTATION_RECIPE_NAMES
from textattack.commands import TextAttackCommand


class AugmentCommand(TextAttackCommand):
    """The TextAttack attack module:

    A command line parser to run data augmentation from user
    specifications.
    """

    def run(self, args):
        """Reads in a CSV, performs augmentation, and outputs an augmented CSV.

        Preserves all columns except for the input (augmneted) column.
        """

        args = textattack.AugmenterArgs(**vars(args))
        if args.interactive:
            print("\nRunning in interactive mode...\n")
            augmenter = eval(AUGMENTATION_RECIPE_NAMES[args.recipe])(
                pct_words_to_swap=args.pct_words_to_swap,
                transformations_per_example=args.transformations_per_example,
                high_yield=args.high_yield,
                fast_augment=args.fast_augment,
                enable_advanced_metrics=args.enable_advanced_metrics,
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
                            recipe_display = " ".join(AUGMENTATION_RECIPE_NAMES.keys())
                            print(f"\n\t{recipe_display}\n")
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

                if args.enable_advanced_metrics:
                    results = augmenter.augment(text)
                    print("Augmentations:\n")
                    for augmentation in results[0]:
                        print(augmentation, "\n")
                    print()
                    print(
                        f"Average Original Perplexity Score: {results[1]['avg_original_perplexity']}"
                    )
                    print(
                        f"Average Augment Perplexity Score: {results[1]['avg_attack_perplexity']}"
                    )
                    print(
                        f"Average Augment USE Score: {results[2]['avg_attack_use_score']}\n"
                    )

                else:
                    for augmentation in augmenter.augment(text):
                        print(augmentation, "\n")
                print("--------------------------------------------------------")
        else:
            textattack.shared.utils.set_seed(args.random_seed)
            start_time = time.time()
            if not (args.input_csv and args.input_column and args.output_csv):
                raise ArgumentError(
                    "The following arguments are required: --csv, --input-column/--i"
                )
            # Validate input/output paths.
            if not os.path.exists(args.input_csv):
                raise FileNotFoundError(f"Can't find CSV at location {args.input_csv}")
            if os.path.exists(args.output_csv):
                if args.overwrite:
                    textattack.shared.logger.info(
                        f"Preparing to overwrite {args.output_csv}."
                    )
                else:
                    raise OSError(
                        f"Outfile {args.output_csv} exists and --overwrite not set."
                    )
            # Read in CSV file as a list of dictionaries. Use the CSV sniffer to
            # try and automatically infer the correct CSV format.
            csv_file = open(args.input_csv, "r")

            # mark where commas and quotes occur within the text value
            def markQuotes(lines):
                for row in lines:
                    row = row.replace('"', '"/')
                    yield row

            dialect = csv.Sniffer().sniff(csv_file.readline(), delimiters=";,")
            csv_file.seek(0)
            rows = [
                row
                for row in csv.DictReader(
                    markQuotes(csv_file),
                    dialect=dialect,
                    skipinitialspace=True,
                )
            ]

            # replace markings with quotations and commas
            for row in rows:
                for item in row:
                    i = 0
                    while i < len(row[item]):
                        if row[item][i] == "/":
                            if row[item][i - 1] == '"':
                                row[item] = row[item][:i] + row[item][i + 1 :]
                            else:
                                row[item] = row[item][:i] + '"' + row[item][i + 1 :]
                        i += 1

            # Validate input column.
            row_keys = set(rows[0].keys())
            if args.input_column not in row_keys:
                raise ValueError(
                    f"Could not find input column {args.input_column} in CSV. Found keys: {row_keys}"
                )
            textattack.shared.logger.info(
                f"Read {len(rows)} rows from {args.input_csv}. Found columns {row_keys}."
            )

            augmenter = eval(AUGMENTATION_RECIPE_NAMES[args.recipe])(
                pct_words_to_swap=args.pct_words_to_swap,
                transformations_per_example=args.transformations_per_example,
                high_yield=args.high_yield,
                fast_augment=args.fast_augment,
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
            with open(args.output_csv, "w") as outfile:
                csv_writer = csv.writer(
                    outfile, delimiter=",", quotechar="/", quoting=csv.QUOTE_MINIMAL
                )
                # Write header.
                csv_writer.writerow(output_rows[0].keys())
                # Write rows.
                for row in output_rows:
                    csv_writer.writerow(row.values())

            textattack.shared.logger.info(
                f"Wrote {len(output_rows)} augmentations to {args.output_csv} in {time.time() - start_time}s."
            )

            # Remove extra markings in output file
            with open(args.output_csv, "r") as file:
                data = file.readlines()
            for i in range(len(data)):
                data[i] = data[i].replace("/", "")
            with open(args.output_csv, "w") as file:
                file.writelines(data)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "augment",
            help="augment text data",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser = textattack.AugmenterArgs._add_parser_args(parser)
        parser.set_defaults(func=AugmentCommand())
