"""

TextAttack CLI main class
==============================

"""


# !/usr/bin/env python
import argparse

from eukaryote.commands.attack_command import AttackCommand
from eukaryote.commands.attack_resume_command import AttackResumeCommand
from eukaryote.commands.augment_command import AugmentCommand
from eukaryote.commands.benchmark_recipe_command import BenchmarkRecipeCommand
from eukaryote.commands.eval_model_command import EvalModelCommand
from eukaryote.commands.list_things_command import ListThingsCommand
from eukaryote.commands.peek_dataset_command import PeekDatasetCommand
from eukaryote.commands.train_model_command import TrainModelCommand

from eukaryote.commands.t4a_attack_eval_command import T4A_AttackEvalCommand
from eukaryote.commands.t4a_attack_train_command import T4A_AttackTrainCommand
from eukaryote.commands.t4a_train_command import T4A_TrainCommand


def main():
    parser = argparse.ArgumentParser(
        "TextAttack CLI",
        usage="[python -m] eukaryote <command> [<args>]",
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

    T4A_AttackEvalCommand.register_subcommand(subparsers)
    T4A_AttackTrainCommand.register_subcommand(subparsers)
    T4A_TrainCommand.register_subcommand(subparsers)

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
