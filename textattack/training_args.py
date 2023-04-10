"""
TrainingArgs Class
==================
"""

from dataclasses import dataclass, field
import datetime
import os
from typing import Union

from textattack.datasets import HuggingFaceDataset
from textattack.models.helpers import LSTMForClassification, WordCNNForClassification
from textattack.models.wrappers import (
    HuggingFaceModelWrapper,
    ModelWrapper,
    PyTorchModelWrapper,
)
from textattack.shared import logger
from textattack.shared.utils import ARGS_SPLIT_TOKEN

from .attack import Attack
from .attack_args import ATTACK_RECIPE_NAMES


def default_output_dir():
    return os.path.join(
        "./outputs", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    )


@dataclass
class TrainingArgs:
    """Arguments for ``Trainer`` class that is used for adversarial training.

    Args:
        num_epochs (:obj:`int`, `optional`, defaults to :obj:`3`):
            Total number of epochs for training.
        num_clean_epochs (:obj:`int`, `optional`, defaults to :obj:`1`):
            Number of epochs to train on just the original training dataset before adversarial training.
        attack_epoch_interval (:obj:`int`, `optional`, defaults to :obj:`1`):
            Generate a new adversarial training set every `N` epochs.
        early_stopping_epochs (:obj:`int`, `optional`, defaults to :obj:`None`):
            Number of epochs validation must increase before stopping early (:obj:`None` for no early stopping).
        learning_rate (:obj:`float`, `optional`, defaults to :obj:`5e-5`):
            Learning rate for optimizer.
        num_warmup_steps (:obj:`int` or :obj:`float`, `optional`, defaults to :obj:`500`):
            The number of steps for the warmup phase of linear scheduler.
            If :obj:`num_warmup_steps` is a :obj:`float` between 0 and 1, the number of warmup steps will be :obj:`math.ceil(num_training_steps * num_warmup_steps)`.
        weight_decay (:obj:`float`, `optional`, defaults to :obj:`0.01`):
            Weight decay (L2 penalty).
        per_device_train_batch_size (:obj:`int`, `optional`, defaults to :obj:`8`):
            The batch size per GPU/CPU for training.
        per_device_eval_batch_size (:obj:`int`, `optional`, defaults to :obj:`32`):
            The batch size per GPU/CPU for evaluation.
        gradient_accumulation_steps (:obj:`int`, `optional`, defaults to :obj:`1`):
            Number of updates steps to accumulate the gradients before performing a backward/update pass.
        random_seed (:obj:`int`, `optional`, defaults to :obj:`786`):
            Random seed for reproducibility.
        parallel (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, train using multiple GPUs using :obj:`torch.DataParallel`.
        load_best_model_at_end (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, keep track of the best model across training and load it at the end.
        alpha (:obj:`float`, `optional`, defaults to :obj:`1.0`):
            The weight for adversarial loss.
        num_train_adv_examples (:obj:`int` or :obj:`float`, `optional`, defaults to :obj:`-1`):
            The number of samples to successfully attack when generating adversarial training set before start of every epoch.
            If :obj:`num_train_adv_examples` is a :obj:`float` between 0 and 1, the number of adversarial examples generated is
            fraction of the original training set.
        query_budget_train (:obj:`int`, `optional`, defaults to :obj:`None`):
            The max query budget to use when generating adversarial training set. :obj:`None` means infinite query budget.
        attack_num_workers_per_device (:obj:`int`, defaults to `optional`, :obj:`1`):
            Number of worker processes to run per device for attack. Same as :obj:`num_workers_per_device` argument for :class:`~textattack.AttackArgs`.
        output_dir (:obj:`str`, `optional`):
            Directory to output training logs and checkpoints. Defaults to :obj:`./outputs/%Y-%m-%d-%H-%M-%S-%f` format.
        checkpoint_interval_steps (:obj:`int`, `optional`, defaults to :obj:`None`):
            If set, save model checkpoint after every `N` updates to the model.
        checkpoint_interval_epochs (:obj:`int`, `optional`, defaults to :obj:`None`):
            If set, save model checkpoint after every `N` epochs.
        save_last (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If :obj:`True`, save the model at end of training. Can be used with :obj:`load_best_model_at_end` to save the best model at the end.
        log_to_tb (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, log to Tensorboard.
        tb_log_dir (:obj:`str`, `optional`, defaults to :obj:`"./runs"`):
            Path of Tensorboard log directory.
        log_to_wandb (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, log to Wandb.
        wandb_project (:obj:`str`, `optional`, defaults to :obj:`"textattack"`):
            Name of Wandb project for logging.
        logging_interval_step (:obj:`int`, `optional`, defaults to :obj:`1`):
            Log to Tensorboard/Wandb every `N` training steps.
    """

    num_epochs: int = 3
    num_clean_epochs: int = 1
    attack_epoch_interval: int = 1
    early_stopping_epochs: int = None
    learning_rate: float = 5e-5
    num_warmup_steps: Union[int, float] = 500
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    random_seed: int = 786
    parallel: bool = False
    load_best_model_at_end: bool = False
    alpha: float = 1.0
    num_train_adv_examples: Union[int, float] = -1
    query_budget_train: int = None
    attack_num_workers_per_device: int = 1
    output_dir: str = field(default_factory=default_output_dir)
    checkpoint_interval_steps: int = None
    checkpoint_interval_epochs: int = None
    save_last: bool = True
    log_to_tb: bool = False
    tb_log_dir: str = None
    log_to_wandb: bool = False
    wandb_project: str = "textattack"
    logging_interval_step: int = 1

    def __post_init__(self):
        assert self.num_epochs > 0, "`num_epochs` must be greater than 0."
        assert (
            self.num_clean_epochs >= 0
        ), "`num_clean_epochs` must be greater than or equal to 0."
        if self.early_stopping_epochs is not None:
            assert (
                self.early_stopping_epochs > 0
            ), "`early_stopping_epochs` must be greater than 0."
        if self.attack_epoch_interval is not None:
            assert (
                self.attack_epoch_interval > 0
            ), "`attack_epoch_interval` must be greater than 0."
        assert (
            self.num_warmup_steps >= 0
        ), "`num_warmup_steps` must be greater than or equal to 0."
        assert (
            self.gradient_accumulation_steps > 0
        ), "`gradient_accumulation_steps` must be greater than 0."
        assert (
            self.num_clean_epochs <= self.num_epochs
        ), f"`num_clean_epochs` cannot be greater than `num_epochs` ({self.num_clean_epochs} > {self.num_epochs})."

        if isinstance(self.num_train_adv_examples, float):
            assert (
                self.num_train_adv_examples >= 0.0
                and self.num_train_adv_examples <= 1.0
            ), "If `num_train_adv_examples` is float, it must be between 0 and 1."
        elif isinstance(self.num_train_adv_examples, int):
            assert (
                self.num_train_adv_examples > 0 or self.num_train_adv_examples == -1
            ), "If `num_train_adv_examples` is int, it must be greater than 0 or equal to -1."
        else:
            raise TypeError(
                "`num_train_adv_examples` must be of either type `int` or `float`."
            )

    @classmethod
    def _add_parser_args(cls, parser):
        """Add listed args to command line parser."""
        default_obj = cls()

        def int_or_float(v):
            try:
                return int(v)
            except ValueError:
                return float(v)

        parser.add_argument(
            "--num-epochs",
            "--epochs",
            type=int,
            default=default_obj.num_epochs,
            help="Total number of epochs for training.",
        )
        parser.add_argument(
            "--num-clean-epochs",
            type=int,
            default=default_obj.num_clean_epochs,
            help="Number of epochs to train on the clean dataset before adversarial training (N/A if --attack unspecified)",
        )
        parser.add_argument(
            "--attack-epoch-interval",
            type=int,
            default=default_obj.attack_epoch_interval,
            help="Generate a new adversarial training set every N epochs.",
        )
        parser.add_argument(
            "--early-stopping-epochs",
            type=int,
            default=default_obj.early_stopping_epochs,
            help="Number of epochs validation must increase before stopping early (-1 for no early stopping)",
        )
        parser.add_argument(
            "--learning-rate",
            "--lr",
            type=float,
            default=default_obj.learning_rate,
            help="Learning rate for Adam Optimization.",
        )
        parser.add_argument(
            "--num-warmup-steps",
            type=int_or_float,
            default=default_obj.num_warmup_steps,
            help="The number of steps for the warmup phase of linear scheduler.",
        )
        parser.add_argument(
            "--weight-decay",
            type=float,
            default=default_obj.weight_decay,
            help="Weight decay (L2 penalty).",
        )
        parser.add_argument(
            "--per-device-train-batch-size",
            type=int,
            default=default_obj.per_device_train_batch_size,
            help="The batch size per GPU/CPU for training.",
        )
        parser.add_argument(
            "--per-device-eval-batch-size",
            type=int,
            default=default_obj.per_device_eval_batch_size,
            help="The batch size per GPU/CPU for evaluation.",
        )
        parser.add_argument(
            "--gradient-accumulation-steps",
            type=int,
            default=default_obj.gradient_accumulation_steps,
            help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            default=default_obj.random_seed,
            help="Random seed.",
        )
        parser.add_argument(
            "--parallel",
            action="store_true",
            default=default_obj.parallel,
            help="If set, run training on multiple GPUs.",
        )
        parser.add_argument(
            "--load-best-model-at-end",
            action="store_true",
            default=default_obj.load_best_model_at_end,
            help="If set, keep track of the best model across training and load it at the end.",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=1.0,
            help="The weight of adversarial loss.",
        )
        parser.add_argument(
            "--num-train-adv-examples",
            type=int_or_float,
            default=default_obj.num_train_adv_examples,
            help="The number of samples to attack when generating adversarial training set. Default is -1 (which is all possible samples).",
        )
        parser.add_argument(
            "--query-budget-train",
            type=int,
            default=default_obj.query_budget_train,
            help="The max query budget to use when generating adversarial training set.",
        )
        parser.add_argument(
            "--attack-num-workers-per-device",
            type=int,
            default=default_obj.attack_num_workers_per_device,
            help="Number of worker processes to run per device for attack. Same as `num_workers_per_device` argument for `AttackArgs`.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=default_output_dir(),
            help="Directory to output training logs and checkpoints.",
        )
        parser.add_argument(
            "--checkpoint-interval-steps",
            type=int,
            default=default_obj.checkpoint_interval_steps,
            help="Save model checkpoint after every N updates to the model.",
        )
        parser.add_argument(
            "--checkpoint-interval-epochs",
            type=int,
            default=default_obj.checkpoint_interval_epochs,
            help="Save model checkpoint after every N epochs.",
        )
        parser.add_argument(
            "--save-last",
            action="store_true",
            default=default_obj.save_last,
            help="If set, save the model at end of training. Can be used with `--load-best-model-at-end` to save the best model at the end.",
        )
        parser.add_argument(
            "--log-to-tb",
            action="store_true",
            default=default_obj.log_to_tb,
            help="If set, log to Tensorboard",
        )
        parser.add_argument(
            "--tb-log-dir",
            type=str,
            default=default_obj.tb_log_dir,
            help="Path of Tensorboard log directory.",
        )
        parser.add_argument(
            "--log-to-wandb",
            action="store_true",
            default=default_obj.log_to_wandb,
            help="If set, log to Wandb.",
        )
        parser.add_argument(
            "--wandb-project",
            type=str,
            default=default_obj.wandb_project,
            help="Name of Wandb project for logging.",
        )
        parser.add_argument(
            "--logging-interval-step",
            type=int,
            default=default_obj.logging_interval_step,
            help="Log to Tensorboard/Wandb every N steps.",
        )

        return parser


@dataclass
class _CommandLineTrainingArgs:
    """Command line interface training args.

    This requires more arguments to create models and get datasets.
    Args:
        model_name_or_path (str): Name or path of the model we want to create. "lstm" and "cnn" will create TextAttack\'s LSTM and CNN models while
            any other input will be used to create Transformers model. (e.g."brt-base-uncased").
        attack (str): Attack recipe to use (enables adversarial training)
        dataset (str): dataset for training; will be loaded from `datasets` library.
        task_type (str): Type of task model is supposed to perform. Options: `classification`, `regression`.
        model_max_length (int): The maximum sequence length of the model.
        model_num_labels (int): The number of labels for classification (1 for regression).
        dataset_train_split (str): Name of the train split. If not provided will try `train` as the split name.
        dataset_eval_split (str): Name of the train split. If not provided will try `dev`, `validation`, or `eval` as split name.
    """

    model_name_or_path: str
    attack: str
    dataset: str
    task_type: str = "classification"
    model_max_length: int = None
    model_num_labels: int = None
    dataset_train_split: str = None
    dataset_eval_split: str = None
    filter_train_by_labels: list = None
    filter_eval_by_labels: list = None

    @classmethod
    def _add_parser_args(cls, parser):
        # Arguments that are needed if we want to create a model to train.
        parser.add_argument(
            "--model-name-or-path",
            "--model",
            type=str,
            required=True,
            help='Name or path of the model we want to create. "lstm" and "cnn" will create TextAttack\'s LSTM and CNN models while'
            ' any other input will be used to create Transformers model. (e.g."brt-base-uncased").',
        )
        parser.add_argument(
            "--model-max-length",
            type=int,
            default=None,
            help="The maximum sequence length of the model.",
        )
        parser.add_argument(
            "--model-num-labels",
            type=int,
            default=None,
            help="The number of labels for classification.",
        )
        parser.add_argument(
            "--attack",
            type=str,
            required=False,
            default=None,
            help="Attack recipe to use (enables adversarial training)",
        )
        parser.add_argument(
            "--task-type",
            type=str,
            default="classification",
            help="Type of task model is supposed to perform. Options: `classification`, `regression`.",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            default="yelp",
            help="dataset for training; will be loaded from "
            "`datasets` library. if dataset has a subset, separate with a colon. "
            " ex: `glue^sst2` or `rotten_tomatoes`",
        )
        parser.add_argument(
            "--dataset-train-split",
            type=str,
            default="",
            help="train dataset split, if non-standard "
            "(can automatically detect 'train'",
        )
        parser.add_argument(
            "--dataset-eval-split",
            type=str,
            default="",
            help="val dataset split, if non-standard "
            "(can automatically detect 'dev', 'validation', 'eval')",
        )
        parser.add_argument(
            "--filter-train-by-labels",
            nargs="+",
            type=int,
            required=False,
            default=None,
            help="List of labels to keep in the train dataset and discard all others.",
        )
        parser.add_argument(
            "--filter-eval-by-labels",
            nargs="+",
            type=int,
            required=False,
            default=None,
            help="List of labels to keep in the eval dataset and discard all others.",
        )
        return parser

    @classmethod
    def _create_model_from_args(cls, args):
        """Given ``CommandLineTrainingArgs``, return specified
        ``textattack.models.wrappers.ModelWrapper`` object."""

        assert isinstance(
            args, cls
        ), f"Expect args to be of type `{type(cls)}`, but got type `{type(args)}`."

        if args.model_name_or_path == "lstm":
            logger.info("Loading textattack model: LSTMForClassification")
            max_seq_len = args.model_max_length if args.model_max_length else 128
            num_labels = args.model_num_labels if args.model_num_labels else 2
            model = LSTMForClassification(
                max_seq_length=max_seq_len,
                num_labels=num_labels,
                emb_layer_trainable=True,
            )
            model = PyTorchModelWrapper(model, model.tokenizer)
        elif args.model_name_or_path == "cnn":
            logger.info("Loading textattack model: WordCNNForClassification")
            max_seq_len = args.model_max_length if args.model_max_length else 128
            num_labels = args.model_num_labels if args.model_num_labels else 2
            model = WordCNNForClassification(
                max_seq_length=max_seq_len,
                num_labels=num_labels,
                emb_layer_trainable=True,
            )
            model = PyTorchModelWrapper(model, model.tokenizer)
        else:
            import transformers

            logger.info(
                f"Loading transformers AutoModelForSequenceClassification: {args.model_name_or_path}"
            )
            max_seq_len = args.model_max_length if args.model_max_length else 512
            num_labels = args.model_num_labels if args.model_num_labels else 2
            config = transformers.AutoConfig.from_pretrained(
                args.model_name_or_path,
                num_labels=num_labels,
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                config=config,
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                model_max_length=max_seq_len,
            )
            model = HuggingFaceModelWrapper(model, tokenizer)

        assert isinstance(
            model, ModelWrapper
        ), "`model` must be of type `textattack.models.wrappers.ModelWrapper`."
        return model

    @classmethod
    def _create_dataset_from_args(cls, args):
        dataset_args = args.dataset.split(ARGS_SPLIT_TOKEN)
        # TODO `HuggingFaceDataset` -> `HuggingFaceDataset`
        if args.dataset_train_split:
            train_dataset = HuggingFaceDataset(
                *dataset_args, split=args.dataset_train_split
            )
        else:
            try:
                train_dataset = HuggingFaceDataset(*dataset_args, split="train")
                args.dataset_train_split = "train"
            except KeyError:
                raise KeyError(
                    f"Error: no `train` split found in `{args.dataset}` dataset"
                )

        if args.dataset_eval_split:
            eval_dataset = HuggingFaceDataset(
                *dataset_args, split=args.dataset_eval_split
            )
        else:
            # try common dev split names
            try:
                eval_dataset = HuggingFaceDataset(*dataset_args, split="dev")
                args.dataset_eval_split = "dev"
            except KeyError:
                try:
                    eval_dataset = HuggingFaceDataset(*dataset_args, split="eval")
                    args.dataset_eval_split = "eval"
                except KeyError:
                    try:
                        eval_dataset = HuggingFaceDataset(
                            *dataset_args, split="validation"
                        )
                        args.dataset_eval_split = "validation"
                    except KeyError:
                        try:
                            eval_dataset = HuggingFaceDataset(
                                *dataset_args, split="test"
                            )
                            args.dataset_eval_split = "test"
                        except KeyError:
                            raise KeyError(
                                f"Could not find `dev`, `eval`, `validation`, or `test` split in dataset {args.dataset}."
                            )

        if args.filter_train_by_labels:
            train_dataset.filter_by_labels_(args.filter_train_by_labels)
        if args.filter_eval_by_labels:
            eval_dataset.filter_by_labels_(args.filter_eval_by_labels)
        # Testing for Coverage of model return values with dataset.
        num_labels = args.model_num_labels if args.model_num_labels else 2

        # Only Perform labels checks if output_column is equal to label.
        if (
            train_dataset.output_column == "label"
            and eval_dataset.output_column == "label"
        ):
            train_dataset_labels = train_dataset._dataset["label"]

            eval_dataset_labels = eval_dataset._dataset["label"]

            train_dataset_labels_set = set(train_dataset_labels)

            assert all(
                label >= 0
                for label in train_dataset_labels_set
                if isinstance(label, int)
            ), f"Train dataset has negative label/s {[label for label in train_dataset_labels_set if isinstance(label,int) and label < 0 ]} which is/are not supported by pytorch.Use --filter-train-by-labels to keep suitable labels"

            assert num_labels >= len(
                train_dataset_labels_set
            ), f"Model constructed has {num_labels} output nodes and train dataset has {len(train_dataset_labels_set)} labels , Model should have output nodes greater than or equal to labels in train dataset.Use --model-num-labels to set model's output nodes."

            eval_dataset_labels_set = set(eval_dataset_labels)

            assert all(
                label >= 0
                for label in eval_dataset_labels_set
                if isinstance(label, int)
            ), f"Eval dataset has negative label/s {[label for label in eval_dataset_labels_set if isinstance(label,int) and label < 0 ]} which is/are not supported by pytorch.Use --filter-eval-by-labels to keep suitable labels"

            assert num_labels >= len(
                set(eval_dataset_labels_set)
            ), f"Model constructed has {num_labels} output nodes and eval dataset has {len(eval_dataset_labels_set)} labels , Model should have output nodes greater than or equal to labels in eval dataset.Use --model-num-labels to set model's output nodes."

        return train_dataset, eval_dataset

    @classmethod
    def _create_attack_from_args(cls, args, model_wrapper):
        import textattack  # noqa: F401

        if args.attack is None:
            return None
        assert (
            args.attack in ATTACK_RECIPE_NAMES
        ), f"Unavailable attack recipe {args.attack}"
        attack = eval(f"{ATTACK_RECIPE_NAMES[args.attack]}.build(model_wrapper)")
        assert isinstance(
            attack, Attack
        ), "`attack` must be of type `textattack.Attack`."
        return attack


# This neat trick allows use to reorder the arguments to avoid TypeErrors commonly found when inheriting dataclass.
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
@dataclass
class CommandLineTrainingArgs(TrainingArgs, _CommandLineTrainingArgs):
    @classmethod
    def _add_parser_args(cls, parser):
        parser = _CommandLineTrainingArgs._add_parser_args(parser)
        parser = TrainingArgs._add_parser_args(parser)
        return parser
