
import time
import tqdm
import torch
import collections

from .attack import Attack
import textattack
from textattack.shared.utils import logger


class AttackArgs:

    def __init__(self):
        pass


class CliAttackArgs(AttackArgs):
    def __init__(self):
        pass

class Attacker:

    def __init__(self, attack, attack_log_manager=None):
        assert isinstance(attack, Attack), f"`attack` argument must be of type `textattack.shared.Attack`, but received argument of type `{type(attack)}`."

        self.attack = attack
        if attack_log_manager is None:
            self.attack_log_manager = textattack.logger.AttackLogManager()
        else:
            self.attack_log_manager = attack_log_manager


    def _attack(self, dataset, attack_args=None, checkpoint=None):
        """Internal method that carries out attack. No parallel processing is involved.

        Args:
            dataset (textattack.datasets.Dataset): Dataset to attack
            attack_args (textattack.shared.AttackArgs, optional): Arguments for attack. This will be overrided by checkpoint's argument if `checkpoint` is not `None`.
            checkpoint (textattack.shared.Checkpoint, optional): Checkpoint from which to resume the attack.
        """
        if checkpoint is not None and attack_args is not None:
            raise ValueError("`attack_args` and `checkpoint` cannot be both set.")
        
        if checkpoint:
            num_remaining_attacks = checkpoint.num_remaining_attacks
            worklist = checkpoint.worklist
            worklist_tail = checkpoint.worklist_tail
            logger.info(
                "Recovered from checkpoint previously saved at {}.".format(
                    checkpoint.datetime
                )
            )
        else:
            num_remaining_attacks = attack_args.num_examples
            worklist = collections.deque(range(0, args.num_examples))
            worklist_tail = worklist[-1]

        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0)
        if checkpoint:
            num_results = checkpoint.results_count
            num_failures = checkpoint.num_failed_attacks
            num_successes = checkpoint.num_successful_attacks
        else:
            num_results = 0
            num_failures = 0
            num_successes = 0

        i = 0
        while i < len(worklist):
            idx = worklist[i]
            i += 1
            example, ground_truth_output = self.dataset[idx]
            example = textattack.shared.AttackedText(example)
            if self.dataset.label_names is not None:
                example.attack_attrs["label_names"] = self.dataset.label_names
            result = self.attack.attack(example, ground_truth_output)
            self.attack_log_manager.log_result(result)

            if not attack_args.disable_stdout:
                print("\n")

            if isinstance(result, textattack.attack_results.SkippedAttackResult) and attack_args.attack_n:
                # `worklist_tail` keeps track of highest idx that has been part of worklist. 
                # This is useful for async-logging that can happen when using parallel processing. 
                # Used to get the next dataset element when attacking with `attack_n` = True.
                worklist_tail += 1
                worklist.append(worklist_tail)
            else:
                pbar.update(1)

            num_results += 1

            if (
                type(result) == textattack.attack_results.SuccessfulAttackResult
                or type(result) == textattack.attack_results.MaximizedAttackResult
            ):
                num_successes += 1
            if type(result) == textattack.attack_results.FailedAttackResult:
                num_failures += 1
            pbar.set_description(
                "[Succeeded / Failed / Total] {} / {} / {}".format(
                    num_successes, num_failures, num_results
                )
            )

            if (
                attack_args.checkpoint_interval
                and len(self.attack_log_manager.results) % attack_args.checkpoint_interval == 0
            ):
                new_checkpoint = textattack.shared.Checkpoint(
                    attack_args, self.attack_log_manager, worklist, worklist_tail
                )
                new_checkpoint.save()
                self.attack_log_manager.flush()

        pbar.close()
        # Enable summary stdout
        if attack_args.disable_stdout:
            self.attack_log_manager.enable_stdout()
        self.attack_log_manager.log_summary()
        self.attack_log_manager.flush()


    def _attack_parallel(self, dataset, num_workers_per_device, attack_args=None, checkpoint=None):
        if checkpoint is not None and attack_args is not None:
            raise ValueError("`attack_args` and `checkpoint` cannot be both set.")
        
        if checkpoint:
            num_remaining_attacks = checkpoint.num_remaining_attacks
            worklist = checkpoint.worklist
            worklist_tail = checkpoint.worklist_tail
            logger.info(
                "Recovered from checkpoint previously saved at {}.".format(
                    checkpoint.datetime
                )
            )
        else:
            num_remaining_attacks = attack_args.num_examples
            worklist = collections.deque(range(0, args.num_examples))
            worklist_tail = worklist[-1]


        # We reserve the first GPU for coordinating workers.
        num_gpus = torch.cuda.device_count()
        num_workers = num_workers_per_device * num_gpus

        textattack.shared.logger.info(f"Running {num_workers} workers on {num_gpus} GPUs")

        in_queue = torch.multiprocessing.Queue()
        out_queue = torch.multiprocessing.Queue()
        # Add stuff to queue.
        missing_datapoints = set()
        for i in worklist:
            try:
                example, ground_truth_output = self.dataset[i]
                example = textattack.shared.AttackedText(example)
                if self.dataset.label_names is not None:
                    example.attack_attrs["label_names"] = self.dataset.label_names
                in_queue.put((i, example, ground_truth_output))
            except IndexError:
                missing_datapoints.add(i)

        # if our dataset is shorter than the number of samples chosen, remove the
        # out-of-bounds indices from the dataset
        for i in missing_datapoints:
            worklist.remove(i)  

    # Start workers.
    torch.multiprocessing.Pool(num_gpus, attack_from_queue, (args, in_queue, out_queue))
    # Log results asynchronously and update progress bar.
    if args.checkpoint_resume:
        num_results = checkpoint.results_count
        num_failures = checkpoint.num_failed_attacks
        num_successes = checkpoint.num_successful_attacks
    else:
        num_results = 0
        num_failures = 0
        num_successes = 0
    pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0)
    while worklist:
        result = out_queue.get(block=True)

        if isinstance(result, Exception):
            raise result
        idx, result = result
        attack_log_manager.log_result(result)
        worklist.remove(idx)
        if (not args.attack_n) or (
            not isinstance(result, textattack.attack_results.SkippedAttackResult)
        ):
            pbar.update()
            num_results += 1

            if (
                type(result) == textattack.attack_results.SuccessfulAttackResult
                or type(result) == textattack.attack_results.MaximizedAttackResult
            ):
                num_successes += 1
            if type(result) == textattack.attack_results.FailedAttackResult:
                num_failures += 1
            pbar.set_description(
                "[Succeeded / Failed / Total] {} / {} / {}".format(
                    num_successes, num_failures, num_results
                )
            )
        else:
            # worklist_tail keeps track of highest idx that has been part of worklist
            # Used to get the next dataset element when attacking with `attack_n` = True.
            worklist_tail += 1
            try:
                text, output = dataset[worklist_tail]
                worklist.append(worklist_tail)
                in_queue.put((worklist_tail, text, output))
            except IndexError:
                raise IndexError(
                    "Tried adding to worklist, but ran out of datapoints. Size of data is {} but tried to access index {}".format(
                        len(dataset), worklist_tail
                    )
                )

        if (
            args.checkpoint_interval
            and len(attack_log_manager.results) % args.checkpoint_interval == 0
        ):
            new_checkpoint = textattack.shared.Checkpoint(
                args, attack_log_manager, worklist, worklist_tail
            )
            new_checkpoint.save()
            attack_log_manager.flush()

    pbar.close()
    print()
    # Enable summary stdout.
    if args.disable_stdout:
        attack_log_manager.enable_stdout()
    attack_log_manager.log_summary()
    attack_log_manager.flush()
    print()

    def attack(self, dataset, attack_args):
        """Attack `dataset` and record results to specified loggers.
        Args:
            dataset (textattack.datasets.Dataset): dataset to attack.
            attack_args (textattack.shared.AttackArgs): arguments for attack.
        """
        assert isinstance(dataset, textattack.datasets.Dataset), f"`dataset` argument must be of type `textattack.datasets.Dataset`, but received argument of type `{type(dataset)}`."
        assert isinstance(dataset, textattack.shared.AttackArgs), f"`attack_args` argument must be of type `textattack.shared.AttackArgs`, but received argument of type `{type(attack_args)}`."

        self._attack(dataset, attack_args=attack_args)

    def attack_parallel(self, dataset, attack_args, num_workers_per_device):
        """Attack `dataset` with single worker and record results to specified loggers.
        Args:
            dataset (textattack.datasets.Dataset): dataset to attack.
            num_workers_per_device (int): Number of worker threads to run per device. For example, if you are using GPUs and ``num_workers_per_device=2``,
                then 2 processes will be running in each GPU. If you are only using CPU, then this is equivalent to running 2 processes concurrently. 
        """
        assert isinstance(dataset, textattack.datasets.Dataset), f"`dataset` argument must be of type `textattack.datasets.Dataset`, but received argument of type `{type(dataset)}`."
        assert isinstance(dataset, textattack.shared.AttackArgs), f"`attack_args` argument must be of type `textattack.shared.AttackArgs`, but received argument of type `{type(attack_args)}`."
    
        self._attack_parallel(dataset, num_workers_per_device, attack_args=attack_args)
        
    def resume_attack(self, dataset, checkpoint):
        """Resume attacking `dataset` from saved `checkpoint`. 
        Args:
            dataset (textattack.datasets.Dataset): dataset to attack.
            checkpoint (textattack.shared.Checkpoint): checkpoint object that has previously been saved.
        """
        assert isinstance(dataset, textattack.datasets.Dataset), f"`dataset` argument must be of type `textattack.datasets.Dataset`, but received argument of type `{type(dataset)}`."
        assert isinstance(dataset, textattack.shared.Checkpoint), f"`checkpoint` argument must be of type `textattack.shared.Checkpoint`, but received argument of type `{type(checkpoint)}`."
        
        self.attack_log_manager = checkpoint.attack_log_manager
        self._attack(dataset, checkpoint=checkpoint)

    def resume_attack_parallel(self, dataset, checkpoint, num_workers_per_device):
        """Resume attacking `dataset` from saved `checkpoint`. 
        Args:
            dataset (textattack.datasets.Dataset): dataset to attack.
            checkpoint (textattack.shared.Checkpoint): checkpoint object that has previously been saved.
        """
        assert isinstance(dataset, textattack.datasets.Dataset), f"`dataset` argument must be of type `textattack.datasets.Dataset`, but received argument of type `{type(dataset)}`."
        assert isinstance(dataset, textattack.shared.Checkpoint), f"`checkpoint` argument must be of type `textattack.shared.Checkpoint`, but received argument of type `{type(checkpoint)}`."
        
        self.attack_log_manager = checkpoint.attack_log_manager
        self._attack(dataset, checkpoint=checkpoint)

