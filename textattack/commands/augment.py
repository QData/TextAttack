from argparse import ArgumentParser

from textattack.commands import TextAttackCommand
class AugmentCommand(TextAttackCommand):
    """
    The TextAttack attack module:
    
        A command line parser to run data augmentation from user specifications.
    """
    
    def run(self):
        raise NotImplementedError('cant benchmark yet')

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser("augment", help="Benchmark a model with TextAttack")