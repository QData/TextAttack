from dataclasses import dataclass

import textattack
from textattack.shared.utils import ARGS_SPLIT_TOKEN, load_module_from_file

HUGGINGFACE_DATASET_BY_MODEL = {
    #
    # bert-base-uncased
    #
    "bert-base-uncased-ag-news": ("ag_news", None, "test"),
    "bert-base-uncased-cola": ("glue", "cola", "validation"),
    "bert-base-uncased-imdb": ("imdb", None, "test"),
    "bert-base-uncased-mnli": ("glue", "mnli", "validation_matched", [1, 2, 0]),
    "bert-base-uncased-mrpc": ("glue", "mrpc", "validation"),
    "bert-base-uncased-qnli": ("glue", "qnli", "validation"),
    "bert-base-uncased-qqp": ("glue", "qqp", "validation"),
    "bert-base-uncased-rte": ("glue", "rte", "validation"),
    "bert-base-uncased-sst2": ("glue", "sst2", "validation"),
    "bert-base-uncased-stsb": ("glue", "stsb", "validation", None, 5.0),
    "bert-base-uncased-wnli": ("glue", "wnli", "validation"),
    "bert-base-uncased-mr": ("rotten_tomatoes", None, "test"),
    "bert-base-uncased-snli": ("snli", None, "test", [1, 2, 0]),
    "bert-base-uncased-yelp": ("yelp_polarity", None, "test"),
    #
    # distilbert-base-cased
    #
    "distilbert-base-cased-cola": ("glue", "cola", "validation"),
    "distilbert-base-cased-mrpc": ("glue", "mrpc", "validation"),
    "distilbert-base-cased-qqp": ("glue", "qqp", "validation"),
    "distilbert-base-cased-snli": ("snli", None, "test"),
    "distilbert-base-cased-sst2": ("glue", "sst2", "validation"),
    "distilbert-base-cased-stsb": ("glue", "stsb", "validation", None, 5.0),
    #
    # distilbert-base-uncased
    #
    "distilbert-base-uncased-ag-news": ("ag_news", None, "test"),
    "distilbert-base-uncased-cola": ("glue", "cola", "validation"),
    "distilbert-base-uncased-imdb": ("imdb", None, "test"),
    "distilbert-base-uncased-mnli": ("glue", "mnli", "validation_matched", [1, 2, 0]),
    "distilbert-base-uncased-mr": ("rotten_tomatoes", None, "test"),
    "distilbert-base-uncased-mrpc": ("glue", "mrpc", "validation"),
    "distilbert-base-uncased-qnli": ("glue", "qnli", "validation"),
    "distilbert-base-uncased-rte": ("glue", "rte", "validation"),
    "distilbert-base-uncased-wnli": ("glue", "wnli", "validation"),
    #
    # roberta-base (RoBERTa is cased by default)
    #
    "roberta-base-ag-news": ("ag_news", None, "test"),
    "roberta-base-cola": ("glue", "cola", "validation"),
    "roberta-base-imdb": ("imdb", None, "test"),
    "roberta-base-mr": ("rotten_tomatoes", None, "test"),
    "roberta-base-mrpc": ("glue", "mrpc", "validation"),
    "roberta-base-qnli": ("glue", "qnli", "validation"),
    "roberta-base-rte": ("glue", "rte", "validation"),
    "roberta-base-sst2": ("glue", "sst2", "validation"),
    "roberta-base-stsb": ("glue", "stsb", "validation", None, 5.0),
    "roberta-base-wnli": ("glue", "wnli", "validation"),
    #
    # albert-base-v2 (ALBERT is cased by default)
    #
    "albert-base-v2-ag-news": ("ag_news", None, "test"),
    "albert-base-v2-cola": ("glue", "cola", "validation"),
    "albert-base-v2-imdb": ("imdb", None, "test"),
    "albert-base-v2-mr": ("rotten_tomatoes", None, "test"),
    "albert-base-v2-rte": ("glue", "rte", "validation"),
    "albert-base-v2-qqp": ("glue", "qqp", "validation"),
    "albert-base-v2-snli": ("snli", None, "test"),
    "albert-base-v2-sst2": ("glue", "sst2", "validation"),
    "albert-base-v2-stsb": ("glue", "stsb", "validation", None, 5.0),
    "albert-base-v2-wnli": ("glue", "wnli", "validation"),
    "albert-base-v2-yelp": ("yelp_polarity", None, "test"),
    #
    # xlnet-base-cased
    #
    "xlnet-base-cased-cola": ("glue", "cola", "validation"),
    "xlnet-base-cased-imdb": ("imdb", None, "test"),
    "xlnet-base-cased-mr": ("rotten_tomatoes", None, "test"),
    "xlnet-base-cased-mrpc": ("glue", "mrpc", "validation"),
    "xlnet-base-cased-rte": ("glue", "rte", "validation"),
    "xlnet-base-cased-stsb": ("glue", "stsb", "validation", None, 5.0),
    "xlnet-base-cased-wnli": ("glue", "wnli", "validation"),
}


#
# Models hosted by textattack.
#
TEXTATTACK_DATASET_BY_MODEL = {
    #
    # LSTMs
    #
    "lstm-ag-news": ("ag_news", None, "test"),
    "lstm-imdb": ("imdb", None, "test"),
    "lstm-mr": ("rotten_tomatoes", None, "test"),
    "lstm-sst2": ("glue", "sst2", "validation"),
    "lstm-yelp": ("yelp_polarity", None, "test"),
    #
    # CNNs
    #
    "cnn-ag-news": ("ag_news", None, "test"),
    "cnn-imdb": ("imdb", None, "test"),
    "cnn-mr": ("rotten_tomatoes", None, "test"),
    "cnn-sst2": ("glue", "sst2", "validation"),
    "cnn-yelp": ("yelp_polarity", None, "test"),
    #
    # T5 for translation
    #
    "t5-en-de": (
        "textattack.datasets.translation.TedMultiTranslationDataset",
        "en",
        "de",
    ),
    "t5-en-fr": (
        "textattack.datasets.translation.TedMultiTranslationDataset",
        "en",
        "fr",
    ),
    "t5-en-ro": (
        "textattack.datasets.translation.TedMultiTranslationDataset",
        "en",
        "de",
    ),
    #
    # T5 for summarization
    #
    "t5-summarization": ("gigaword", None, "test"),
}


@dataclass
class DatasetArgs:
    """Arguments for loading model from command line input."""

    dataset_from_huggingface: str
    dataset_from_file: str
    shuffle: bool

    @classmethod
    def add_parser_args(cls, parser):
        """Adds dataset-related arguments to an argparser.

        This is useful because we want to load pretrained models using
        multiple different parsers that share these, but not all,
        arguments.
        """
        dataset_group = parser.add_mutually_exclusive_group()
        dataset_group.add_argument(
            "--dataset-from-huggingface",
            type=str,
            required=False,
            default=None,
            help="Dataset to load from `datasets` repository.",
        )
        dataset_group.add_argument(
            "--dataset-from-file",
            type=str,
            required=False,
            default=None,
            help="Dataset to load from a file.",
        )
        parser.add_argument(
            "--shuffle",
            type=eval,
            required=False,
            choices=[True, False],
            default="True",
            help="Randomly shuffle the data before attacking",
        )
        return parser

    @classmethod
    def parse_dataset_from_args(cls, args):
        assert isinstance(
            args, cls
        ), f"Expect args to be of type `{type(cls)}`, but got type `{type(args)}`."

        # Automatically detect dataset for huggingface & textattack models.
        # This allows us to use the --model shortcut without specifying a dataset.
        if args.model in HUGGINGFACE_DATASET_BY_MODEL:
            args.dataset_from_huggingface = HUGGINGFACE_DATASET_BY_MODEL[args.model]
        elif args.model in TEXTATTACK_DATASET_BY_MODEL:
            dataset = TEXTATTACK_DATASET_BY_MODEL[args.model]
            if dataset[0].startswith("textattack"):
                # unsavory way to pass custom dataset classes
                # ex: dataset = ('textattack.datasets.translation.TedMultiTranslationDataset', 'en', 'de')
                dataset = eval(f"{dataset[0]}")(*dataset[1:])
                if args.shuffle:
                    dataset.shuffle()
                return dataset
            else:
                args.dataset_from_huggingface = dataset

        # Get dataset from args.
        if args.dataset_from_file:
            textattack.shared.logger.info(
                f"Loading model and tokenizer from file: {args.model_from_file}"
            )
            if ARGS_SPLIT_TOKEN in args.dataset_from_file:
                dataset_file, dataset_name = args.dataset_from_file.split(
                    ARGS_SPLIT_TOKEN
                )
            else:
                dataset_file, dataset_name = args.dataset_from_file, "dataset"
            try:
                dataset_module = load_module_from_file(dataset_file)
            except Exception:
                raise ValueError(f"Failed to import file {args.dataset_from_file}")
            try:
                dataset = getattr(dataset_module, dataset_name)
                if args.shuffle:
                    dataset.shuffle()
            except AttributeError:
                raise AttributeError(
                    f"Variable ``dataset`` not found in module {args.dataset_from_file}"
                )
        elif args.dataset_from_huggingface:
            dataset_args = args.dataset_from_huggingface
            if isinstance(dataset_args, str):
                if ARGS_SPLIT_TOKEN in dataset_args:
                    dataset_args = dataset_args.split(ARGS_SPLIT_TOKEN)
                else:
                    dataset_args = (dataset_args,)
            dataset = textattack.datasets.HuggingFaceDataset(
                *dataset_args, shuffle=args.shuffle
            )
        else:
            raise ValueError("Must supply pretrained model or dataset")

        assert isinstance(
            dataset, textattack.datasets.Dataset
        ), "Loaded `dataset` must be of type `textattack.datasets.Dataset`."
        return dataset
