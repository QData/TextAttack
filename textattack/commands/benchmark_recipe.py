"""

TextAttack benchmark recipe Command
=====================================

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from textattack.commands import TextAttackCommand


class BenchmarkRecipeCommand(TextAttackCommand):
    """The TextAttack benchmark recipe module:

    A command line parser to benchmark a recipe from user
    specifications.
    """

    def run(self, args):
        raise NotImplementedError("Cannot benchmark recipes yet - stay tuned!!")

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "benchmark-recipe",
            help="benchmark a recipe",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.set_defaults(func=BenchmarkRecipeCommand())
