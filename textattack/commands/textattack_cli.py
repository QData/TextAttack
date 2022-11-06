"""

TextAttack CLI main class
==============================

"""


# !/usr/bin/env python
import argparse

from textattack.commands.attack_command import AttackCommand
from textattack.commands.attack_resume_command import AttackResumeCommand
from textattack.commands.augment_command import AugmentCommand
from textattack.commands.benchmark_recipe_command import BenchmarkRecipeCommand
from textattack.commands.eval_model_command import EvalModelCommand
from textattack.commands.list_things_command import ListThingsCommand
from textattack.commands.peek_dataset_command import PeekDatasetCommand
from textattack.commands.train_model_command import TrainModelCommand


def main():
    parser = argparse.ArgumentParser(
        "TextAttack CLI",
        usage="[python -m] textattack <command> [<args>]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(help="textattack command helpers")

    # Register commands
    AttackCommand.register_subcommand(subparsers)
    AttackResumeCommand.register_subcommand(subparsers)
    AugmentCommand.register_subcommand(subparsers)
    BenchmarkRecipeCommand.register_subcommand(subparsers)
    EvalModelCommand.register_subcommand(subparsers)
    ListThingsCommand.register_subcommand(subparsers)
    TrainModelCommand.register_subcommand(subparsers)
    PeekDatasetCommand.register_subcommand(subparsers)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    func = args.func
    del args.func
    func.run(args)


if __name__ == "__main__":
    main()
