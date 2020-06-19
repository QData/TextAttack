from argparse import ArgumentParser

import textattack
from textattack.commands import TextAttackCommand

from textattack.commands.attack.attack_args import *
from textattack.commands.attack.attack_args_helpers import *


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

class AttackCommand(TextAttackCommand):
    """
    The TextAttack attack module:
    
        A command line parser to run an attack from user specifications.
    """
    
    def run(self):
        from textattack.commands.attack.run_attack_parallel import run as run_parallel
        from textattack.commands.attack.run_attack_single_threaded import (
            run as run_single_threaded,
        )
        if self.parallel:
            run_parallel(self)
        else:
            run_single_threaded(self)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser("attack", help="Run an attack on an NLP model")
        transformation_names = set(BLACK_BOX_TRANSFORMATION_CLASS_NAMES.keys()) | set(
            WHITE_BOX_TRANSFORMATION_CLASS_NAMES.keys()
        )
        parser.add_argument(
            "--transformation",
            type=str,
            required=False,
            default="word-swap-embedding",
            choices=transformation_names,
            help='The transformation to apply. Usage: "--transformation {transformation}:{arg_1}={value_1},{arg_3}={value_3}. Choices: '
            + str(transformation_names),
        )
    
        model_group = parser.add_mutually_exclusive_group()
    
        model_names = list(HUGGINGFACE_DATASET_BY_MODEL.keys()) + list(
            TEXTATTACK_DATASET_BY_MODEL.keys()
        )
        model_group.add_argument(
            "--model",
            type=str,
            required=False,
            default=None,
            choices=model_names,
            help="The pre-trained model to attack.",
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
            help="huggingface.co ID of pre-trained model to load",
        )
    
        dataset_group = parser.add_mutually_exclusive_group()
        dataset_group.add_argument(
            "--dataset-from-nlp",
            type=str,
            required=False,
            default=None,
            help="Dataset to load from `nlp` repository.",
        )
        dataset_group.add_argument(
            "--dataset-from-file",
            type=str,
            required=False,
            default=None,
            help="Dataset to load from a file.",
        )
    
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
            "--out-dir",
            type=str,
            required=False,
            default=None,
            help="A directory to output results to.",
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
            "--enable-csv",
            nargs="?",
            default=None,
            const="fancy",
            type=str,
            help="Enable logging to csv. Use --enable-csv plain to remove [[]] around words.",
        )
    
        parser.add_argument(
            "--num-examples",
            "-n",
            type=int,
            required=False,
            default="5",
            help="The number of examples to process.",
        )
    
        parser.add_argument(
            "--num-examples-offset",
            "-o",
            type=int,
            required=False,
            default=0,
            help="The offset to start at in the dataset.",
        )
    
        parser.add_argument(
            "--shuffle",
            action="store_true",
            required=False,
            default=False,
            help="Randomly shuffle the data before attacking",
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
    
        parser.add_argument("--random-seed", default=str_to_int("TEXTATTACK"))
    
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
    
        attack_group = parser.add_mutually_exclusive_group(required=False)
        search_choices = ", ".join(SEARCH_CLASS_NAMES.keys())
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
            choices=RECIPE_NAMES.keys(),
        )
        attack_group.add_argument(
            "--attack-from-file",
            type=str,
            required=False,
            default=None,
            help="attack to load from file (overrides provided goal function, transformation & constraints)",
        )
    
        # Parser for parsing args for resume
        resume_parser = argparse.ArgumentParser(
            description="A commandline parser for TextAttack",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        resume_parser.add_argument(
            "--checkpoint-file",
            "-f",
            type=str,
            required=True,
            help='Path of checkpoint file to resume attack from. If "latest" (or "{directory path}/latest") is entered,'
            "recover latest checkpoint from either current path or specified directory.",
        )
    
        resume_parser.add_argument(
            "--checkpoint-dir",
            "-d",
            required=False,
            type=str,
            default=None,
            help="The directory to save checkpoint files. If not set, use directory from recovered arguments.",
        )
    
        resume_parser.add_argument(
            "--checkpoint-interval",
            "-i",
            required=False,
            type=int,
            help="If set, checkpoint will be saved after attacking every N examples. If not set, no checkpoints will be saved.",
        )
    
        resume_parser.add_argument(
            "--parallel",
            action="store_true",
            default=False,
            help="Run attack using multiple GPUs.",
        )
    
        # Resume attack from checkpoint.
        if sys.argv[1:] and sys.argv[1].lower() == "resume":
            args = resume_parser.parse_args(sys.argv[2:])
            setattr(args, "checkpoint_resume", True)
        else:
            command_line_args = (
                None if sys.argv[1:] else ["-h"]
            )  # Default to help with empty arguments.
            args = parser.parse_args(command_line_args)
            setattr(args, "checkpoint_resume", False)
    
            if args.checkpoint_interval and args.shuffle:
                # Not allowed b/c we cannot recover order of shuffled data
                raise ValueError("Cannot use `--checkpoint-interval` with `--shuffle=True`")
    
            set_seed(args.random_seed)
    
        # Shortcuts for huggingface models using --model.
        if not args.checkpoint_resume and args.model in HUGGINGFACE_DATASET_BY_MODEL:
            _, args.dataset_from_nlp = HUGGINGFACE_DATASET_BY_MODEL[args.model]
        elif not args.checkpoint_resume and args.model in TEXTATTACK_DATASET_BY_MODEL:
            _, args.dataset_from_nlp = TEXTATTACK_DATASET_BY_MODEL[args.model]
    
        return args
    
    