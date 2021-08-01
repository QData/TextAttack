"""
ModelArgs Class
===============
"""


from dataclasses import dataclass
import json
import os

import transformers

import textattack
from textattack.shared.utils import ARGS_SPLIT_TOKEN, load_module_from_file

HUGGINGFACE_MODELS = {
    #
    # bert-base-uncased
    #
    "bert-base-uncased": "bert-base-uncased",
    "bert-base-uncased-ag-news": "textattack/bert-base-uncased-ag-news",
    "bert-base-uncased-cola": "textattack/bert-base-uncased-CoLA",
    "bert-base-uncased-imdb": "textattack/bert-base-uncased-imdb",
    "bert-base-uncased-mnli": "textattack/bert-base-uncased-MNLI",
    "bert-base-uncased-mrpc": "textattack/bert-base-uncased-MRPC",
    "bert-base-uncased-qnli": "textattack/bert-base-uncased-QNLI",
    "bert-base-uncased-qqp": "textattack/bert-base-uncased-QQP",
    "bert-base-uncased-rte": "textattack/bert-base-uncased-RTE",
    "bert-base-uncased-sst2": "textattack/bert-base-uncased-SST-2",
    "bert-base-uncased-stsb": "textattack/bert-base-uncased-STS-B",
    "bert-base-uncased-wnli": "textattack/bert-base-uncased-WNLI",
    "bert-base-uncased-mr": "textattack/bert-base-uncased-rotten-tomatoes",
    "bert-base-uncased-snli": "textattack/bert-base-uncased-snli",
    "bert-base-uncased-yelp": "textattack/bert-base-uncased-yelp-polarity",
    #
    # distilbert-base-cased
    #
    "distilbert-base-uncased": "distilbert-base-uncased",
    "distilbert-base-cased-cola": "textattack/distilbert-base-cased-CoLA",
    "distilbert-base-cased-mrpc": "textattack/distilbert-base-cased-MRPC",
    "distilbert-base-cased-qqp": "textattack/distilbert-base-cased-QQP",
    "distilbert-base-cased-snli": "textattack/distilbert-base-cased-snli",
    "distilbert-base-cased-sst2": "textattack/distilbert-base-cased-SST-2",
    "distilbert-base-cased-stsb": "textattack/distilbert-base-cased-STS-B",
    "distilbert-base-uncased-ag-news": "textattack/distilbert-base-uncased-ag-news",
    "distilbert-base-uncased-cola": "textattack/distilbert-base-cased-CoLA",
    "distilbert-base-uncased-imdb": "textattack/distilbert-base-uncased-imdb",
    "distilbert-base-uncased-mnli": "textattack/distilbert-base-uncased-MNLI",
    "distilbert-base-uncased-mr": "textattack/distilbert-base-uncased-rotten-tomatoes",
    "distilbert-base-uncased-mrpc": "textattack/distilbert-base-uncased-MRPC",
    "distilbert-base-uncased-qnli": "textattack/distilbert-base-uncased-QNLI",
    "distilbert-base-uncased-rte": "textattack/distilbert-base-uncased-RTE",
    "distilbert-base-uncased-wnli": "textattack/distilbert-base-uncased-WNLI",
    #
    # roberta-base (RoBERTa is cased by default)
    #
    "roberta-base": "roberta-base",
    "roberta-base-ag-news": "textattack/roberta-base-ag-news",
    "roberta-base-cola": "textattack/roberta-base-CoLA",
    "roberta-base-imdb": "textattack/roberta-base-imdb",
    "roberta-base-mr": "textattack/roberta-base-rotten-tomatoes",
    "roberta-base-mrpc": "textattack/roberta-base-MRPC",
    "roberta-base-qnli": "textattack/roberta-base-QNLI",
    "roberta-base-rte": "textattack/roberta-base-RTE",
    "roberta-base-sst2": "textattack/roberta-base-SST-2",
    "roberta-base-stsb": "textattack/roberta-base-STS-B",
    "roberta-base-wnli": "textattack/roberta-base-WNLI",
    #
    # albert-base-v2 (ALBERT is cased by default)
    #
    "albert-base-v2": "albert-base-v2",
    "albert-base-v2-ag-news": "textattack/albert-base-v2-ag-news",
    "albert-base-v2-cola": "textattack/albert-base-v2-CoLA",
    "albert-base-v2-imdb": "textattack/albert-base-v2-imdb",
    "albert-base-v2-mr": "textattack/albert-base-v2-rotten-tomatoes",
    "albert-base-v2-rte": "textattack/albert-base-v2-RTE",
    "albert-base-v2-qqp": "textattack/albert-base-v2-QQP",
    "albert-base-v2-snli": "textattack/albert-base-v2-snli",
    "albert-base-v2-sst2": "textattack/albert-base-v2-SST-2",
    "albert-base-v2-stsb": "textattack/albert-base-v2-STS-B",
    "albert-base-v2-wnli": "textattack/albert-base-v2-WNLI",
    "albert-base-v2-yelp": "textattack/albert-base-v2-yelp-polarity",
    #
    # xlnet-base-cased
    #
    "xlnet-base-cased": "xlnet-base-cased",
    "xlnet-base-cased-cola": "textattack/xlnet-base-cased-CoLA",
    "xlnet-base-cased-imdb": "textattack/xlnet-base-cased-imdb",
    "xlnet-base-cased-mr": "textattack/xlnet-base-cased-rotten-tomatoes",
    "xlnet-base-cased-mrpc": "textattack/xlnet-base-cased-MRPC",
    "xlnet-base-cased-rte": "textattack/xlnet-base-cased-RTE",
    "xlnet-base-cased-stsb": "textattack/xlnet-base-cased-STS-B",
    "xlnet-base-cased-wnli": "textattack/xlnet-base-cased-WNLI",
}


#
# Models hosted by textattack.
# `models` vs `models_v2`: `models_v2` is simply a new dir in S3 that contains models' `config.json`.
# Fixes issue https://github.com/QData/TextAttack/issues/485
# Model parameters has not changed.
#
TEXTATTACK_MODELS = {
    #
    # LSTMs
    #
    "lstm-ag-news": "models_v2/classification/lstm/ag-news",
    "lstm-imdb": "models_v2/classification/lstm/imdb",
    "lstm-mr": "models_v2/classification/lstm/mr",
    "lstm-sst2": "models_v2/classification/lstm/sst2",
    "lstm-yelp": "models_v2/classification/lstm/yelp",
    #
    # CNNs
    #
    "cnn-ag-news": "models_v2/classification/cnn/ag-news",
    "cnn-imdb": "models_v2/classification/cnn/imdb",
    "cnn-mr": "models_v2/classification/cnn/rotten-tomatoes",
    "cnn-sst2": "models_v2/classification/cnn/sst",
    "cnn-yelp": "models_v2/classification/cnn/yelp",
    #
    # T5 for translation
    #
    "t5-en-de": "english_to_german",
    "t5-en-fr": "english_to_french",
    "t5-en-ro": "english_to_romanian",
    #
    # T5 for summarization
    #
    "t5-summarization": "summarization",
}


@dataclass
class ModelArgs:
    """Arguments for loading base/pretrained or trained models."""

    model: str = None
    model_from_file: str = None
    model_from_huggingface: str = None

    @classmethod
    def _add_parser_args(cls, parser):
        """Adds model-related arguments to an argparser."""
        model_group = parser.add_mutually_exclusive_group()

        model_names = list(HUGGINGFACE_MODELS.keys()) + list(TEXTATTACK_MODELS.keys())
        model_group.add_argument(
            "--model",
            type=str,
            required=False,
            default=None,
            help="Name of or path to a pre-trained TextAttack model to load. Choices: "
            + str(model_names),
        )
        model_group.add_argument(
            "--model-from-file",
            type=str,
            required=False,
            help="File of model and tokenizer to import.",
        )
        model_group.add_argument(
            "--model-from-huggingface",
            type=str,
            required=False,
            help="Name of or path of pre-trained HuggingFace model to load.",
        )

        return parser

    @classmethod
    def _create_model_from_args(cls, args):
        """Given ``ModelArgs``, return specified
        ``textattack.models.wrappers.ModelWrapper`` object."""

        assert isinstance(
            args, cls
        ), f"Expect args to be of type `{type(cls)}`, but got type `{type(args)}`."

        if args.model_from_file:
            # Support loading the model from a .py file where a model wrapper
            # is instantiated.
            colored_model_name = textattack.shared.utils.color_text(
                args.model_from_file, color="blue", method="ansi"
            )
            textattack.shared.logger.info(
                f"Loading model and tokenizer from file: {colored_model_name}"
            )
            if ARGS_SPLIT_TOKEN in args.model_from_file:
                model_file, model_name = args.model_from_file.split(ARGS_SPLIT_TOKEN)
            else:
                _, model_name = args.model_from_file, "model"
            try:
                model_module = load_module_from_file(args.model_from_file)
            except Exception:
                raise ValueError(f"Failed to import file {args.model_from_file}.")
            try:
                model = getattr(model_module, model_name)
            except AttributeError:
                raise AttributeError(
                    f"Variable `{model_name}` not found in module {args.model_from_file}."
                )

            if not isinstance(model, textattack.models.wrappers.ModelWrapper):
                raise TypeError(
                    f"Variable `{model_name}` must be of type "
                    f"``textattack.models.ModelWrapper``, got type {type(model)}."
                )
        elif (args.model in HUGGINGFACE_MODELS) or args.model_from_huggingface:
            # Support loading models automatically from the HuggingFace model hub.

            model_name = (
                HUGGINGFACE_MODELS[args.model]
                if (args.model in HUGGINGFACE_MODELS)
                else args.model_from_huggingface
            )
            colored_model_name = textattack.shared.utils.color_text(
                model_name, color="blue", method="ansi"
            )
            textattack.shared.logger.info(
                f"Loading pre-trained model from HuggingFace model repository: {colored_model_name}"
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, use_fast=True
            )
            model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        elif args.model in TEXTATTACK_MODELS:
            # Support loading TextAttack pre-trained models via just a keyword.
            colored_model_name = textattack.shared.utils.color_text(
                args.model, color="blue", method="ansi"
            )
            if args.model.startswith("lstm"):
                textattack.shared.logger.info(
                    f"Loading pre-trained TextAttack LSTM: {colored_model_name}"
                )
                model = textattack.models.helpers.LSTMForClassification.from_pretrained(
                    args.model
                )
            elif args.model.startswith("cnn"):
                textattack.shared.logger.info(
                    f"Loading pre-trained TextAttack CNN: {colored_model_name}"
                )
                model = (
                    textattack.models.helpers.WordCNNForClassification.from_pretrained(
                        args.model
                    )
                )
            elif args.model.startswith("t5"):
                model = textattack.models.helpers.T5ForTextToText.from_pretrained(
                    args.model
                )
            else:
                raise ValueError(f"Unknown textattack model {args.model}")

            # Choose the approprate model wrapper (based on whether or not this is
            # a HuggingFace model).
            if isinstance(model, textattack.models.helpers.T5ForTextToText):
                model = textattack.models.wrappers.HuggingFaceModelWrapper(
                    model, model.tokenizer
                )
            else:
                model = textattack.models.wrappers.PyTorchModelWrapper(
                    model, model.tokenizer
                )
        elif args.model and os.path.exists(args.model):
            # Support loading TextAttack-trained models via just their folder path.
            # If `args.model` is a path/directory, let's assume it was a model
            # trained with textattack, and try and load it.
            if os.path.exists(os.path.join(args.model, "t5-wrapper-config.json")):
                model = textattack.models.helpers.T5ForTextToText.from_pretrained(
                    args.model
                )
                model = textattack.models.wrappers.HuggingFaceModelWrapper(
                    model, model.tokenizer
                )
            elif os.path.exists(os.path.join(args.model, "config.json")):
                with open(os.path.join(args.model, "config.json")) as f:
                    config = json.load(f)
                model_class = config["architectures"]
                if (
                    model_class == "LSTMForClassification"
                    or model_class == "WordCNNForClassification"
                ):
                    model = eval(
                        f"textattack.models.helpers.{model_class}.from_pretrained({args.model})"
                    )
                    model = textattack.models.wrappers.PyTorchModelWrapper(
                        model, model.tokenizer
                    )
                else:
                    # assume the model is from HuggingFace.
                    model = (
                        transformers.AutoModelForSequenceClassification.from_pretrained(
                            args.model
                        )
                    )
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        args.model, use_fast=True
                    )
                    model = textattack.models.wrappers.HuggingFaceModelWrapper(
                        model, tokenizer
                    )
        else:
            raise ValueError(f"Error: unsupported TextAttack model {args.model}")

        assert isinstance(
            model, textattack.models.wrappers.ModelWrapper
        ), "`model` must be of type `textattack.models.wrappers.ModelWrapper`."
        return model
