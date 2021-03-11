from dataclasses import dataclass, field
import datetime
import os

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

# TODO Add `metric_for_best_model` argument. Currently we just use accuracy for classification and MSE for regression by default.


def default_output_dir():
    return os.path.join(
        "./outputs", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    )


@dataclass
class TrainingArgs:
    """Args for TextAttack ``Trainer`` class that is used for running
    adversarial training.

    Args:
        num_epochs (int): Total number of epochs for training. Default is 5.
        num_clean_epochs (int): Number of epochs to train on just the original training dataset before adversarial training. Default is 0.
        attack_epoch_interval (int): Generate a new adversarial training set every N epochs. Default is 1.
        early_stopping_epochs (int): Number of epochs validation must increase before stopping early (-1 for no early stopping). Default is `None`.
        learning_rate (float): Learning rate for Adam Optimization. Default is 2e-5.
        num_warmup_steps (int): The number of steps for the warmup phase of linear scheduler. Default is 500.
        weight_decay (float): Weight decay (L2 penalty). Default is 0.01.
        per_device_train_batch_size (int): The batch size per GPU/CPU for training. Default is 8.
        per_device_eval_batch_size (int): The batch size per GPU/CPU for evaluation. Default is 32.
        gradient_accumulation_steps (int): Number of updates steps to accumulate the gradients for, before performing a backward/update pass. Default is 1.
        random_seed (int): Random seed. Default is 786.
        parallel (bool): If `True`, train using multiple GPUs. Default is `False`.
        load_best_model_at_end (bool): If `True`, keep track of the best model across training and load it at the end.
        eval_adversarial_robustness (bool): If set, evaluate adversarial robustness on evaluation dataset after every epoch.
        num_eval_adv_examples (int): The number of samples attack if `eval_adversarial_robustness=True`. Default is 1000.
        num_train_adv_examples (int): The number of samples to attack when generating adversarial training set. Default is -1 (which is all possible samples).
        query_budget_train (:obj:`int`, optional): The max query budget to use when generating adversarial training set.
        query_budget_eval (:obj:`int`, optional): The max query budget to use when evaluating adversarial robustness.
        attack_num_workers_per_device (int): Number of worker processes to run per device for attack. Same as `num_workers_per_device` argument for `AttackArgs`.
        output_dir (str): Directory to output training logs and checkpoints.
        checkpoint_interval_steps (int): Save model checkpoint after every N updates to the model.
        checkpoint_interval_epochs (int): Save model checkpoint after every N epochs
        save_last (bool): If `True`, save the model at end of training. Can be used with `load_best_model_at_end` to save the best model at the end. Default is `True`.
        log_to_tb (bool): If `True`, log to Tensorboard. Default is `False`
        tb_log_dir (str): Path of Tensorboard log directory.
        log_to_wandb (bool): If `True`, log to Wandb. Default is `False`.
        wandb_project (str): Name of Wandb project for logging. Default is `textattack`.
        logging_interval_step (int): Log to Tensorboard/Wandb every N steps.
    """

    num_epochs: int = 5
    num_clean_epochs: int = 0
    attack_epoch_interval: int = 1
    early_stopping_epochs: int = None
    learning_rate: float = 5e-5
    lr: float = None  # alternative keyword arg for learning_rate
    num_warmup_steps: int = 500
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    random_seed: int = 786
    parallel: bool = False
    load_best_model_at_end: bool = False
    eval_adversarial_robustness: bool = False
    num_eval_adv_examples: int = 1000
    num_train_adv_examples: int = -1
    query_budget_train: int = None
    query_budget_eval: int = None
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
        if self.lr:
            self.learning_rate = self.lr
        assert self.num_epochs > 0, "`num_epochs` must be greater than 0."
        assert self.num_clean_epochs >= 0, "`num_clean_epochs` must be greater than or equal to 0."
        if self.early_stopping_epochs is not None:
            assert self.early_stopping_epochs > 0, "`early_stopping_epochs` must be greater than 0."
        assert self.attack_epoch_interval > 0, "`attack_epoch_interval` must be greater than 0."
        assert self.num_warmup_steps > 0, "`num_warmup_steps` must be greater than 0."
        assert self.num_train_adv_examples > 0 or self.num_train_adv_examples == -1, "`num_train_adv_examples` must be greater than 0 or equal to -1."
        assert self.num_eval_adv_examples > 0 or self.num_eval_adv_examples == -1, "`num_eval_adv_examples` must be greater than 0 or equal to -1."
        assert self.gradient_accumulation_steps > 0, "`gradient_accumulation_steps` must be greater than 0."
        assert self.num_clean_epochs <= self.num_epochs, f"`num_clean_epochs` cannot be greater than `num_epochs` ({self.num_clean_epochs} > {self.num_epochs})."


    @classmethod
    def add_parser_args(cls, parser):
        """Add listed args to command line parser."""
        parser.add_argument(
            "--num-epochs",
            type=int,
            default=4,
            help="Total number of epochs for training.",
        )
        parser.add_argument(
            "--num-clean-epochs",
            type=int,
            default=0,
            help="Number of epochs to train on the clean dataset before adversarial training (N/A if --attack unspecified)",
        )
        parser.add_argument(
            "--attack-epoch-interval",
            type=int,
            default=1,
            help="Generate a new adversarial training set every N epochs.",
        )
        parser.add_argument(
            "--early-stopping-epochs",
            type=int,
            default=None,
            help="Number of epochs validation must increase before stopping early (-1 for no early stopping)",
        )
        parser.add_argument(
            "--learning-rate",
            "--lr",
            type=float,
            default=5e-5,
            help="Learning rate for Adam Optimization.",
        )
        parser.add_argument(
            "--num-warmup-steps",
            type=float,
            default=500,
            help="The number of steps for the warmup phase of linear scheduler.",
        )
        parser.add_argument(
            "--weight-decay",
            type=float,
            default=0.01,
            help="Weight decay (L2 penalty).",
        )
        parser.add_argument(
            "--per-device-train-batch-size",
            type=int,
            default=8,
            help="The batch size per GPU/CPU for training.",
        )
        parser.add_argument(
            "--per-device-eval-batch-size",
            type=int,
            default=32,
            help="The batch size per GPU/CPU for evaluation.",
        )
        parser.add_argument(
            "--gradient-accumulation-steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.",
        )
        parser.add_argument("--random-seed", type=int, default=786, help="Random seed.")
        parser.add_argument(
            "--parallel",
            action="store_true",
            default=False,
            help="If set, run training on multiple GPUs.",
        )
        parser.add_argument(
            "--load-best-model-at-end",
            action="store_true",
            default=False,
            help="If set, keep track of the best model across training and load it at the end.",
        )
        parser.add_argument(
            "--eval-adversarial-robustness",
            action="store_true",
            default=False,
            help="If set, evaluate adversarial robustness on evaluation dataset after every epoch.",
        )
        parser.add_argument(
            "--num-eval-adv-examples",
            type=int,
            default=1000,
            help="The number of samples attack if `eval_adversarial_robustness=True`. Default is 1000.",
        )
        parser.add_argument(
            "--num-train-adv-examples",
            type=int,
            default=-1,
            help="The number of samples to attack when generating adversarial training set. Default is -1 (which is all possible samples).",
        )
        parser.add_argument(
            "--query-budget-train",
            type=int,
            default=None,
            help="The max query budget to use when generating adversarial training set.",
        )
        parser.add_argument(
            "--query-budget-eval",
            type=int,
            default=None,
            help="The max query budget to use when evaluating adversarial robustness.",
        )
        parser.add_argument(
            "--attack-num-workers-per-device",
            type=int,
            default=1,
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
            default=None,
            help="Save model checkpoint after every N updates to the model.",
        )
        parser.add_argument(
            "--checkpoint-interval-epochs",
            type=int,
            default=None,
            help="Save model checkpoint after every N epochs.",
        )
        parser.add_argument(
            "--save-last",
            action="store_true",
            default=True,
            help="If set, save the model at end of training. Can be used with `--load-best-model-at-end` to save the best model at the end.",
        )
        parser.add_argument(
            "--log-to-tb",
            action="store_true",
            default=False,
            help="If set, log to Tensorboard",
        )
        parser.add_argument(
            "--tb-log-dir",
            type=str,
            default=None,
            help="Path of Tensorboard log directory.",
        )
        parser.add_argument(
            "--log-to-wandb",
            action="store_true",
            default=False,
            help="If set, log to Wandb.",
        )
        parser.add_argument(
            "--wandb-project",
            type=str,
            default="textattack",
            help="Name of Wandb project for logging.",
        )
        parser.add_argument(
            "--logging-interval-step",
            type=int,
            default=1,
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

    @classmethod
    def add_parser_args(cls, parser):
        # Arguments that are needed if we want to create a model to train.
        parser.add_argument(
            "--model-name-or-path",
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
            required=True,
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
        return parser

    @classmethod
    def create_model_from_args(cls, args):
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
                max_position_embeddings=max_seq_len,
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
    def create_dataset_from_args(cls, args):
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

        return train_dataset, eval_dataset

    @classmethod
    def create_attack_from_args(cls, args, model_wrapper):
        import textattack

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
    def add_parser_args(cls, parser):
        parser = _CommandLineTrainingArgs.add_parser_args(parser)
        parser = TrainingArgs.add_parser_args(parser)
        return parser
