#!/usr/bin/env python
import argparse

from textattack.commands.attack import AttackCommand
from textattack.commands.augment import AugmentCommand
from textattack.commands.benchmark_model import BenchmarkModelCommand
from textattack.commands.benchmark_recipe import BenchmarkRecipeCommand

def main():
    parser = argparse.ArgumentParser("TextAttack CLI", usage="[python -m] texattack <command> [<args>]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    commands_parser = parser.add_subparsers(help="transformers-cli command helpers")

    # Register commands
    AttackCommand.register_subcommand(commands_parser)
    AugmentCommand.register_subcommand(commands_parser)
    BenchmarkModelCommand.register_subcommand(commands_parser)
    BenchmarkRecipeCommand.register_subcommand(commands_parser)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
