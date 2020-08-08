from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import datetime
import os

from textattack.commands import TextAttackCommand


class TrainModelCommand(TextAttackCommand):
    """The TextAttack train module:

    A command line parser to train a model from user specifications.
    """

    def run(self, args):

        date_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        current_dir = os.path.dirname(os.path.realpath(__file__))
        outputs_dir = os.path.join(
            current_dir, os.pardir, os.pardir, os.pardir, "outputs", "training"
        )
        outputs_dir = os.path.normpath(outputs_dir)

        args.output_dir = os.path.join(
            outputs_dir, f"{args.model}-{args.dataset}-{date_now}/"
        )

        from .run_training import train_model

        train_model(args)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "train",
            help="train a model for sequence classification",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--model", type=str, required=True, help="directory of model to train",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            default="yelp",
            help="dataset for training; will be loaded from "
            "`nlp` library. if dataset has a subset, separate with a colon. "
            " ex: `glue^sst2` or `rotten_tomatoes`",
        )
        parser.add_argument(
            "--pct-dataset",
            type=float,
            default=1.0,
            help="Fraction of dataset to use during training ([0., 1.])",
        )
        parser.add_argument(
            "--dataset-train-split",
            "--train-split",
            type=str,
            default="",
            help="train dataset split, if non-standard "
            "(can automatically detect 'train'",
        )
        parser.add_argument(
            "--dataset-dev-split",
            "--dataset-eval-split",
            "--dev-split",
            type=str,
            default="",
            help="val dataset split, if non-standard "
            "(can automatically detect 'dev', 'validation', 'eval')",
        )
        parser.add_argument(
            "--tb-writer-step",
            type=int,
            default=1,
            help="Number of steps before writing to tensorboard",
        )
        parser.add_argument(
            "--checkpoint-steps",
            type=int,
            default=-1,
            help="save model after this many steps (-1 for no checkpointing)",
        )
        parser.add_argument(
            "--checkpoint-every-epoch",
            action="store_true",
            default=False,
            help="save model checkpoint after each epoch",
        )
        parser.add_argument(
            "--save-last",
            action="store_true",
            default=False,
            help="Overwrite the saved model weights after the final epoch.",
        )
        parser.add_argument(
            "--num-train-epochs",
            "--epochs",
            type=int,
            default=100,
            help="Total number of epochs to train for",
        )
        parser.add_argument(
            "--attack",
            type=str,
            default=None,
            help="Attack recipe to use (enables adversarial training)",
        )
        parser.add_argument(
            "--check-robustness",
            default=False,
            action="store_true",
            help="run attack each epoch to measure robustness, but train normally",
        )
        parser.add_argument(
            "--num-clean-epochs",
            type=int,
            default=1,
            help="Number of epochs to train on the clean dataset before adversarial training (N/A if --attack unspecified)",
        )
        parser.add_argument(
            "--attack-period",
            type=int,
            default=1,
            help="How often (in epochs) to generate a new adversarial training set (N/A if --attack unspecified)",
        )
        parser.add_argument(
            "--augment", type=str, default=None, help="Augmentation recipe to use",
        )
        parser.add_argument(
            "--pct-words-to-swap",
            type=float,
            default=0.1,
            help="Percentage of words to modify when using data augmentation (--augment)",
        )
        parser.add_argument(
            "--transformations-per-example",
            type=int,
            default=4,
            help="Number of augmented versions to create from each example when using data augmentation (--augment)",
        )
        parser.add_argument(
            "--allowed-labels",
            type=int,
            nargs="*",
            default=[],
            help="Labels allowed for training (examples with other labels will be discarded)",
        )
        parser.add_argument(
            "--early-stopping-epochs",
            type=int,
            default=-1,
            help="Number of epochs validation must increase"
            " before stopping early (-1 for no early stopping)",
        )
        parser.add_argument(
            "--batch-size", type=int, default=128, help="Batch size for training"
        )
        parser.add_argument(
            "--max-length",
            type=int,
            default=512,
            help="Maximum length of a sequence (anything beyond this will "
            "be truncated)",
        )
        parser.add_argument(
            "--learning-rate",
            "--lr",
            type=float,
            default=2e-5,
            help="Learning rate for Adam Optimization",
        )
        parser.add_argument(
            "--grad-accum-steps",
            type=int,
            default=1,
            help="Number of steps to accumulate gradients before optimizing, "
            "advancing scheduler, etc.",
        )
        parser.add_argument(
            "--warmup-proportion",
            type=float,
            default=0.1,
            help="Warmup proportion for linear scheduling",
        )
        parser.add_argument(
            "--config-name",
            type=str,
            default="config.json",
            help="Filename to save BERT config as",
        )
        parser.add_argument(
            "--weights-name",
            type=str,
            default="pytorch_model.bin",
            help="Filename to save model weights as",
        )
        parser.add_argument(
            "--enable-wandb",
            default=False,
            action="store_true",
            help="log metrics to Weights & Biases",
        )
        parser.add_argument("--random-seed", default=21, type=int)

        parser.set_defaults(func=TrainModelCommand())
