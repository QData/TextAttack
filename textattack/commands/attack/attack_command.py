from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import textattack
from textattack.commands import TextAttackCommand
from textattack.commands.attack.attack_args import (
    ATTACK_RECIPE_NAMES,
    BLACK_BOX_TRANSFORMATION_CLASS_NAMES,
    CONSTRAINT_CLASS_NAMES,
    GOAL_FUNCTION_CLASS_NAMES,
    SEARCH_METHOD_CLASS_NAMES,
    WHITE_BOX_TRANSFORMATION_CLASS_NAMES,
)
from textattack.commands.attack.attack_args_helpers import (
    add_dataset_args,
    add_model_args,
    default_checkpoint_dir,
)


class AttackCommand(TextAttackCommand):
    """The TextAttack attack module:

    A command line parser to run an attack from user specifications.
    """

    def run(self, args):
        if args.checkpoint_interval and args.shuffle:
            # Not allowed b/c we cannot recover order of shuffled data
            raise ValueError("Cannot use `--checkpoint-interval` with `--shuffle=True`")

        textattack.shared.utils.set_seed(args.random_seed)
        args.checkpoint_resume = False

        from textattack.commands.attack.run_attack_parallel import run as run_parallel
        from textattack.commands.attack.run_attack_single_threaded import (
            run as run_single_threaded,
        )

        if args.parallel:
            run_parallel(args)
        else:
            run_single_threaded(args)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "attack",
            help="run an attack on an NLP model",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        transformation_names = set(BLACK_BOX_TRANSFORMATION_CLASS_NAMES.keys()) | set(
            WHITE_BOX_TRANSFORMATION_CLASS_NAMES.keys()
        )
        parser.add_argument(
            "--transformation",
            type=str,
            required=False,
            default="word-swap-embedding",
            help='The transformation to apply. Usage: "--transformation {transformation}:{arg_1}={value_1},{arg_3}={value_3}". Choices: '
            + str(transformation_names),
        )

        add_model_args(parser)
        add_dataset_args(parser)

        parser.add_argument(
            "--constraints",
            type=str,
            required=False,
            nargs="*",
            default=["repeat", "stopword"],
            help='Constraints to add to the attack. Usage: "--constraints {constraint}:{arg_1}={value_1},{arg_3}={value_3}". Choices: '
            + str(CONSTRAINT_CLASS_NAMES.keys()),
        )

        parser.add_argument(
            "--log-to-txt",
            "-l",
            nargs="?",
            default=None,
            const="",
            type=str,
            help="Save attack logs to <install-dir>/outputs/~ by default; Include '/' at the end of argument to save "
            "output to specified directory in default naming convention; otherwise enter argument to specify "
            "file name",
        )

        parser.add_argument(
            "--log-to-csv",
            nargs="?",
            default=None,
            const="",
            type=str,
            help="Save attack logs to <install-dir>/outputs/~ by default; Include '/' at the end of argument to save "
            "output to specified directory in default naming convention; otherwise enter argument to specify "
            "file name",
        )

        parser.add_argument(
            "--csv-style",
            default=None,
            const="fancy",
            nargs="?",
            type=str,
            help="Use --csv-style plain to remove [[]] around words",
        )

        parser.add_argument(
            "--enable-visdom", action="store_true", help="Enable logging to visdom."
        )

        parser.add_argument(
            "--enable-wandb",
            action="store_true",
            help="Enable logging to Weights & Biases.",
        )

        parser.add_argument(
            "--disable-stdout", action="store_true", help="Disable logging to stdout"
        )

        parser.add_argument(
            "--interactive",
            action="store_true",
            default=False,
            help="Whether to run attacks interactively.",
        )

        parser.add_argument(
            "--attack-n",
            action="store_true",
            default=False,
            help="Whether to run attack until `n` examples have been attacked (not skipped).",
        )

        parser.add_argument(
            "--parallel",
            action="store_true",
            default=False,
            help="Run attack using multiple GPUs.",
        )

        goal_function_choices = ", ".join(GOAL_FUNCTION_CLASS_NAMES.keys())
        parser.add_argument(
            "--goal-function",
            "-g",
            default="untargeted-classification",
            help=f"The goal function to use. choices: {goal_function_choices}",
        )

        def str_to_int(s):
            return sum((ord(c) for c in s))

        parser.add_argument("--random-seed", default=str_to_int("TEXTATTACK"), type=int)

        parser.add_argument(
            "--checkpoint-dir",
            required=False,
            type=str,
            default=default_checkpoint_dir(),
            help="The directory to save checkpoint files.",
        )

        parser.add_argument(
            "--checkpoint-interval",
            required=False,
            type=int,
            help="If set, checkpoint will be saved after attacking every N examples. If not set, no checkpoints will be saved.",
        )

        parser.add_argument(
            "--query-budget",
            "-q",
            type=int,
            default=float("inf"),
            help="The maximum number of model queries allowed per example attacked.",
        )
        parser.add_argument(
            "--model-batch-size",
            type=int,
            default=32,
            help="The batch size for making calls to the model.",
        )
        parser.add_argument(
            "--model-cache-size",
            type=int,
            default=2 ** 18,
            help="The maximum number of items to keep in the model results cache at once.",
        )
        parser.add_argument(
            "--constraint-cache-size",
            type=int,
            default=2 ** 18,
            help="The maximum number of items to keep in the constraints cache at once.",
        )

        attack_group = parser.add_mutually_exclusive_group(required=False)
        search_choices = ", ".join(SEARCH_METHOD_CLASS_NAMES.keys())
        attack_group.add_argument(
            "--search",
            "--search-method",
            "-s",
            type=str,
            required=False,
            default="greedy-word-wir",
            help=f"The search method to use. choices: {search_choices}",
        )
        attack_group.add_argument(
            "--recipe",
            "--attack-recipe",
            "-r",
            type=str,
            required=False,
            default=None,
            help="full attack recipe (overrides provided goal function, transformation & constraints)",
            choices=ATTACK_RECIPE_NAMES.keys(),
        )
        attack_group.add_argument(
            "--attack-from-file",
            type=str,
            required=False,
            default=None,
            help="attack to load from file (overrides provided goal function, transformation & constraints)",
        )

        parser.set_defaults(func=AttackCommand())
