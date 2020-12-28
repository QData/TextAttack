from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from textattack import Attacker
from textattack.args import CommandLineAttackArgs, DatasetArgs, ModelArgs
from textattack.commands import TextAttackCommand


class AttackCommand(TextAttackCommand):
    """The TextAttack attack module:

    A command line parser to run an attack from user specifications.
    """

    def run(self, args):
        attack_args = CommandLineAttackArgs(**dict(args))
        model_wrapper = ModelArgs.parse_model_from_args(attack_args)
        dataset = DatasetArgs.parse_dataset_from_args(attack_args)

        attack = CommandLineAttackArgs.parse_attack_from_args(
            attack_args, model_wrapper
        )
        attack_log_manager = CommandLineAttackArgs.parse_logger_from_args(attack_args)

        attacker = Attacker(attack, attack_log_manager=attack_log_manager)
        if attack_args.parallel:
            attacker.attack_parallel(dataset, attack_args, num_workers_per_device=1)
        else:
            attacker.attack(dataset, attack_args)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "attack",
            help="run an attack on an NLP model",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser = CommandLineAttackArgs.add_parser_args(parser)
        parser.set_defaults(func=AttackCommand())
