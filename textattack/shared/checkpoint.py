import copy
import datetime
import os
import pickle
import time

from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from textattack.shared import logger, utils


class Checkpoint:
    """An object that stores necessary information for saving and loading
    checkpoints.

    Args:
        args: Command line arguments of the original attack
        log_manager (AttackLogManager): Object for storing attack results
        worklist (deque[int]): List of examples that will be attacked. Examples are represented by their indicies within the dataset.
        worklist_tail (int): Highest index that had been in the worklist at any given time. Used to get the next dataset element
            when attacking with `attack_n` = True.
        chkpt_time (float): epoch time representing when checkpoint was made
    """

    def __init__(self, args, log_manager, worklist, worklist_tail, chkpt_time=None):
        self.args = copy.deepcopy(args)
        self.log_manager = log_manager
        self.worklist = worklist
        self.worklist_tail = worklist_tail
        if chkpt_time:
            self.time = chkpt_time
        else:
            self.time = time.time()

        self._verify()

    def __repr__(self):
        main_str = "Checkpoint("
        lines = []
        lines.append(utils.add_indent(f"(Time):  {self.datetime}", 2))

        args_lines = []
        recipe_set = (
            True
            if "recipe" in self.args.__dict__ and self.args.__dict__["recipe"]
            else False
        )
        mutually_exclusive_args = ["search", "transformation", "constraints", "recipe"]
        if recipe_set:
            args_lines.append(
                utils.add_indent(f'(recipe): {self.args.__dict__["recipe"]}', 2)
            )
        else:
            args_lines.append(
                utils.add_indent(f'(search): {self.args.__dict__["search"]}', 2)
            )
            args_lines.append(
                utils.add_indent(
                    f'(transformation): {self.args.__dict__["transformation"]}', 2
                )
            )
            args_lines.append(
                utils.add_indent(
                    f'(constraints): {self.args.__dict__["constraints"]}', 2
                )
            )

        for key in self.args.__dict__:
            if key not in mutually_exclusive_args:
                args_lines.append(
                    utils.add_indent(f"({key}): {self.args.__dict__[key]}", 2)
                )

        args_str = utils.add_indent("\n" + "\n".join(args_lines), 2)
        lines.append(utils.add_indent(f"(Args):  {args_str}", 2))

        attack_logger_lines = []
        attack_logger_lines.append(
            utils.add_indent(
                f"(Total number of examples to attack): {self.args.num_examples}", 2
            )
        )
        attack_logger_lines.append(
            utils.add_indent(f"(Number of attacks performed): {self.results_count}", 2)
        )
        attack_logger_lines.append(
            utils.add_indent(
                f"(Number of remaining attacks): {self.num_remaining_attacks}", 2
            )
        )
        breakdown_lines = []
        breakdown_lines.append(
            utils.add_indent(
                f"(Number of successful attacks): {self.num_successful_attacks}", 2
            )
        )
        breakdown_lines.append(
            utils.add_indent(
                f"(Number of failed attacks): {self.num_failed_attacks}", 2
            )
        )
        breakdown_lines.append(
            utils.add_indent(
                f"(Number of maximized attacks): {self.num_maximized_attacks}", 2
            )
        )
        breakdown_lines.append(
            utils.add_indent(
                f"(Number of skipped attacks): {self.num_skipped_attacks}", 2
            )
        )
        breakdown_str = utils.add_indent("\n" + "\n".join(breakdown_lines), 2)
        attack_logger_lines.append(
            utils.add_indent(f"(Latest result breakdown): {breakdown_str}", 2)
        )
        attack_logger_str = utils.add_indent("\n" + "\n".join(attack_logger_lines), 2)
        lines.append(
            utils.add_indent(f"(Previous attack summary):  {attack_logger_str}", 2)
        )

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    __str__ = __repr__

    @property
    def results_count(self):
        """Return number of attacks made so far."""
        return len(self.log_manager.results)

    @property
    def num_skipped_attacks(self):
        return sum(isinstance(r, SkippedAttackResult) for r in self.log_manager.results)

    @property
    def num_failed_attacks(self):
        return sum(isinstance(r, FailedAttackResult) for r in self.log_manager.results)

    @property
    def num_successful_attacks(self):
        return sum(
            isinstance(r, SuccessfulAttackResult) for r in self.log_manager.results
        )

    @property
    def num_maximized_attacks(self):
        return sum(
            isinstance(r, MaximizedAttackResult) for r in self.log_manager.results
        )

    @property
    def num_remaining_attacks(self):
        if self.args.attack_n:
            non_skipped_attacks = self.num_successful_attacks + self.num_failed_attacks
            count = self.args.num_examples - non_skipped_attacks
        else:
            count = self.args.num_examples - self.results_count
        return count

    @property
    def dataset_offset(self):
        """Calculate offset into the dataset to start from."""
        # Original offset + # of results processed so far
        return self.args.num_examples_offset + self.results_count

    @property
    def datetime(self):
        return datetime.datetime.fromtimestamp(self.time).strftime("%Y-%m-%d %H:%M:%S")

    def save(self, quiet=False):
        file_name = "{}.ta.chkpt".format(int(self.time * 1000))
        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        path = os.path.join(self.args.checkpoint_dir, file_name)
        if not quiet:
            print("\n\n" + "=" * 125)
            logger.info(
                'Saving checkpoint under "{}" at {} after {} attacks.'.format(
                    path, self.datetime, self.results_count
                )
            )
            print("=" * 125 + "\n")
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(self, path):
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        assert isinstance(checkpoint, Checkpoint)

        return checkpoint

    def _verify(self):
        """Check that the checkpoint has no duplicates and is consistent."""
        assert self.num_remaining_attacks == len(
            self.worklist
        ), "Recorded number of remaining attacks and size of worklist are different."

        results_set = set()
        for result in self.log_manager.results:
            results_set.add(result.original_text)

        assert len(results_set) == self.results_count, "Duplicate AttackResults found."
