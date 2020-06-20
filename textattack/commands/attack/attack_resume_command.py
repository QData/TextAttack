from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from textattack.commands import TextAttackCommand


class AttackResumeCommand(TextAttackCommand):
    """
    The TextAttack attack resume recipe module:
    
        A command line parser to resume a checkpointed attack from user specifications.
    """

    def run(self):
        textattack.shared.utils.set_seed(self.random_seed)
        self.checkpoint_resume = True

        # Run attack from checkpoint.
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
