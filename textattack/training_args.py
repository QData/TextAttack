from dataclasses import dataclass, field
import datetime
import os

# TODO Add `metric_for_best_model` argument. Currently we just use accuracy for classification and MSE for regression by default.


@dataclass
class TrainingArgs:
    """Args for TextAttack ``Trainer`` class that is used for running
    adversarial training.

    Args:
        num_epochs (int): Total number of epochs for training. Default is 5.
        num_clean_epochs (int): Number of epochs to train on just the original training dataset before adversarial training. Default is 1.
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
        output_dir (str): Directory to output training logs and checkpoints.
        checkpoint_interval_steps (int): Save model checkpoint after every N updates to the model.
        checkpoint_interval_epochs (int): Save model checkpoint after every N epochs
        save_last (bool): If `True`, save the model at end of training. Can be used with `load_best_model_at_end` to save the best model at the end. Default is `True`.
        log_to_tb (bool): If `True`, log to Tensorboard. Default is `False`
        tb_log_dir (str): Path of Tensorboard log directory.
        log_to_wandb (bool): If `True`, log to Wandb. Default is `False`.
        wand_project (str): Name of Wandb project for logging. Default is `textattack`.
        logging_interval_step (int): Log to Tensorboard/Wandb every N steps.
    """

    num_epochs: int = 5
    num_clean_epochs: int = 1
    attack_epoch_interval: int = 1
    early_stopping_epochs: int = None
    learning_rate: float = 2e-5
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
    output_dir: str = field(
        default_factory=lambda: os.path.join(
            "./outputs", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        )
    )
    checkpoint_interval_steps: int = None
    checkpoint_interval_epochs: int = None
    save_last: bool = True
    log_to_tb: bool = False
    tb_log_dir: str = None
    log_to_wandb: bool = False
    wand_project: str = "textattack"
    logging_interval_step: int = 1

    @classmethod
    def add_parser_args(cls, parser):
        """Add listed args to command line parser."""
        parser.add_argument(
            "--num-epochs",
            "--epochs",
            type=int,
            default=10,
            help="Total number of epochs for training.",
        )
        parser.add_argument(
            "--num-clean-epochs",
            type=int,
            default=4,
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
            default=2e-5,
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
        parser.add_arguments(
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
        parser.add_arguments(
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
            "--checkpoint-interval-steps",
            type=int,
            default=None,
            help="Save model checkpoint after every N updates to the model.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
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
