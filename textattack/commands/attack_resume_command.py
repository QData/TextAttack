"""

AttackResumeCommand class
===========================

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

import textattack
from textattack import Attacker, CommandLineAttackArgs, DatasetArgs, ModelArgs
from textattack.commands import TextAttackCommand


class AttackResumeCommand(TextAttackCommand):
    """The TextAttack attack resume recipe module:

    A command line parser to resume a checkpointed attack from user
    specifications.
    """

    def run(self, args):
        checkpoint = self._parse_checkpoint_from_args(args)
        assert isinstance(checkpoint.attack_args, CommandLineAttackArgs), (
            f"Expect `attack_args` to be of type `textattack.args.CommandLineAttackArgs`, but got type `{type(checkpoint.attack_args)}`. "
            f"If saved `attack_args` is not of type `textattack.args.CommandLineAttackArgs`, cannot resume attack from command line."
        )
        # merge/update arguments
        checkpoint.attack_args.parallel = args.parallel
        if args.checkpoint_dir:
            checkpoint.attack_args.checkpoint_dir = args.checkpoint_dir
        if args.checkpoint_interval:
            checkpoint.attack_args.checkpoint_interval = args.checkpoint_interval

        model_wrapper = ModelArgs._create_model_from_args(
            checkpoint.attack_args.attack_args
        )
        attack = CommandLineAttackArgs._create_attack_from_args(
            checkpoint.attack_args, model_wrapper
        )
        dataset = DatasetArgs.parse_dataset_from_args(checkpoint.attack_args)
        attacker = Attacker.from_checkpoint(attack, dataset, checkpoint)
        attacker.attack_dataset()

    def _parse_checkpoint_from_args(self, args):
        file_name = os.path.basename(args.checkpoint_file)
        if file_name.lower() == "latest":
            dir_path = os.path.dirname(args.checkpoint_file)
            dir_path = dir_path if dir_path else "."
            chkpt_file_names = [
                f for f in os.listdir(dir_path) if f.endswith(".ta.chkpt")
            ]
            assert chkpt_file_names, "AttackCheckpoint directory is empty"
            timestamps = [int(f.replace(".ta.chkpt", "")) for f in chkpt_file_names]
            latest_file = str(max(timestamps)) + ".ta.chkpt"
            checkpoint_path = os.path.join(dir_path, latest_file)
        else:
            checkpoint_path = args.checkpoint_file

        checkpoint = textattack.shared.AttackCheckpoint.load(checkpoint_path)

        return checkpoint

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        resume_parser = main_parser.add_parser(
            "attack-resume",
            help="resume a checkpointed attack",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )

        # Parser for parsing args for resume
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

        resume_parser.set_defaults(func=AttackResumeCommand())
