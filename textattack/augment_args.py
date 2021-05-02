from dataclasses import dataclass

AUGMENTATION_RECIPE_NAMES = {
    "wordnet": "textattack.augmentation.WordNetAugmenter",
    "embedding": "textattack.augmentation.EmbeddingAugmenter",
    "charswap": "textattack.augmentation.CharSwapAugmenter",
    "eda": "textattack.augmentation.EasyDataAugmenter",
    "checklist": "textattack.augmentation.CheckListAugmenter",
    "clare": "textattack.augmentation.CLAREAugmenter",
}


@dataclass
class AugmenterArgs:
    """Arguments for performing data augmentation.

    Args:
        input_csv (str): Path of input CSV file to augment.
        output_csv (str): Path of CSV file to output augmented data.
    """

    input_csv: str
    output_csv: str
    input_column: str
    recipe: str = "embedding"
    pct_words_to_swap: float = 0.1
    transformations_per_example: int = 2
    random_seed: int = 42
    exclude_original: bool = False
    overwrite: bool = False
    interactive: bool = False

    @classmethod
    def _add_parser_args(cls, parser):
        parser.add_argument(
            "--input-csv",
            required=True,
            type=str,
            help="Path of input CSV file to augment.",
        )
        parser.add_argument(
            "--output-csv",
            required=True,
            type=str,
            help="Path of CSV file to output augmented data.",
        )
        parser.add_argument(
            "--input-column",
            "--i",
            required=True,
            type=str,
            help="CSV input column to be augmented",
        )
        parser.add_argument(
            "--recipe",
            "-r",
            help="Name of augmentation recipe",
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
            "--random-seed", default=42, type=int, help="random seed to set"
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

        return parser
