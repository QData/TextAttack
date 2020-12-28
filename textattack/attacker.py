import collections
import os

import torch
import tqdm

import textattack
from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from textattack.shared.utils import logger

from .attack import Attack


class Attacker:
    """Class for running attacks on a dataset with specified parameters. This
    class uses the ``textattack.shared.Attack`` to actually run the attacks,
    while also providing useful features such as parallel processing,
    saving/resuming from a checkpint, logging to files and stdout.

    Args:
        attack (textattack.Attack): Attack object used to run an attack. It is composed of a goal function, transformation, set of constraints, and search method.
        attack_log_manager (textattack.loggers.AttackLogManager): Object that manages loggers to log to text files, CSV files, Wandb, etc. If not set, the default is stdout output.
            Note that this is replaced when resuming attack from a saved ``AttackCheckpoint``.
    """

    def __init__(self, attack, attack_log_manager=None):
        assert isinstance(
            attack, Attack
        ), f"`attack` argument must be of type `textattack.Attack`, but got type of `{type(attack)}`."

        self.attack = attack
        if attack_log_manager is None:
            self.attack_log_manager = textattack.logger.AttackLogManager()
            self.attack_log_manager.enable_stdout()
        else:
            self.attack_log_manager = attack_log_manager

    def _attack(self, dataset, attack_args=None, checkpoint=None):
        """Internal method that carries out attack. No parallel processing is
        involved.

        Args:
            dataset (textattack.datasets.Dataset): Dataset to attack
            attack_args (textattack.args.AttackArgs, optional): Arguments for attack. This will be overrided by checkpoint's argument if `checkpoint` is not `None`.
            checkpoint (textattack.shared.AttackCheckpoint, optional): AttackCheckpoint from which to resume the attack.
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
            worklist = collections.deque(range(0, attack_args.num_examples))
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
        while worklist:
            idx = worklist.popleft()
            i += 1
            example, ground_truth_output = dataset[idx]
            example = textattack.shared.AttackedText(example)
            if dataset.label_names is not None:
                example.attack_attrs["label_names"] = dataset.label_names
            result = self.attack.attack(example, ground_truth_output)
            self.attack_log_manager.log_result(result)

            if not attack_args.disable_stdout:
                print("\n")

            if isinstance(result, SkippedAttackResult) and attack_args.attack_n:
                # `worklist_tail` keeps track of highest idx that has been part of worklist.
                # This is useful for async-logging that can happen when using parallel processing.
                # Used to get the next dataset element when attacking with `attack_n` = True.
                worklist_tail += 1
                worklist.append(worklist_tail)
            else:
                pbar.update(1)

            num_results += 1

            if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
                num_successes += 1
            if isinstance(result, FailedAttackResult):
                num_failures += 1
            pbar.set_description(
                f"[Succeeded / Failed / Total] {num_successes} / {num_failures} / {num_results}"
            )

            if (
                attack_args.checkpoint_interval
                and len(self.attack_log_manager.results)
                % attack_args.checkpoint_interval
                == 0
            ):
                new_checkpoint = textattack.shared.AttackCheckpoint(
                    attack_args, self.attack_log_manager, worklist, worklist_tail
                )
                new_checkpoint.save()
                self.attack_log_manager.flush()

        pbar.close()
        print()
        # Enable summary stdout
        if attack_args.disable_stdout:
            self.attack_log_manager.enable_stdout()
        self.attack_log_manager.log_summary()
        self.attack_log_manager.flush()
        print()

    def _attack_parallel(
        self, dataset, num_workers_per_device, attack_args=None, checkpoint=None
    ):
        if checkpoint is not None and attack_args is not None:
            raise ValueError("`attack_args` and `checkpoint` cannot be both set.")

        pytorch_multiprocessing_workaround()

        if checkpoint:
            num_remaining_attacks = checkpoint.num_remaining_attacks
            worklist = checkpoint.worklist
            worklist_tail = checkpoint.worklist_tail
            logger.info(
                f"Recovered from checkpoint previously saved at {checkpoint.datetime}."
            )
        else:
            num_remaining_attacks = attack_args.num_examples
            worklist = collections.deque(range(0, attack_args.num_examples))
            worklist_tail = worklist[-1]

        in_queue = torch.multiprocessing.Queue()
        out_queue = torch.multiprocessing.Queue()
        for i in worklist:
            try:
                example, ground_truth_output = dataset[i]
                example = textattack.shared.AttackedText(example)
                if dataset.label_names is not None:
                    example.attack_attrs["label_names"] = dataset.label_names
                in_queue.put((i, example, ground_truth_output))
            except IndexError:
                raise IndexError(
                    f"Tried to access element at {i} in dataset of size {len(dataset)}."
                )

        # We reserve the first GPU for coordinating workers.
        num_gpus = torch.cuda.device_count()
        num_workers = num_workers_per_device * num_gpus
        logger.info(f"Running {num_workers} workers on {num_gpus} GPUs")

        # Start workers.
        torch.multiprocessing.Pool(
            num_workers,
            attack_from_queue,
            (attack_args, self.attack, in_queue, out_queue),
        )
        # Log results asynchronously and update progress bar.
        if checkpoint:
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
            self.attack_log_manager.log_result(result)
            worklist.remove(idx)
            if attack_args.attack_n and isinstance(result, SkippedAttackResult):
                # worklist_tail keeps track of highest idx that has been part of worklist
                # Used to get the next dataset element when attacking with `attack_n` = True.
                worklist_tail += 1
                try:
                    example, ground_truth_output = dataset[worklist_tail]
                    example = textattack.shared.AttackedText(example)
                    if dataset.label_names is not None:
                        example.attack_attrs["label_names"] = dataset.label_names
                    worklist.append(worklist_tail)
                    in_queue.put((worklist_tail, example, ground_truth_output))
                except IndexError:
                    logger.warn(
                        f"Attempted to attack {attack_args.num_examples} examples with but ran out of examples. "
                        f"You might see fewer number of results than {attack_args.num_examples}."
                    )
            else:
                pbar.update()
                num_results += 1

                if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
                    num_successes += 1
                if isinstance(result, FailedAttackResult):
                    num_failures += 1
                pbar.set_description(
                    f"[Succeeded / Failed / Total] {num_successes} / {num_failures} / {num_results}"
                )

            if (
                attack_args.checkpoint_interval
                and len(self.attack_log_manager.results)
                % attack_args.checkpoint_interval
                == 0
            ):
                new_checkpoint = textattack.shared.AttackCheckpoint(
                    attack_args, self.attack_log_manager, worklist, worklist_tail
                )
                new_checkpoint.save()
                self.attack_log_manager.flush()

        pbar.close()
        print()
        # Enable summary stdout.
        if attack_args.disable_stdout:
            self.attack_log_manager.enable_stdout()
        self.attack_log_manager.log_summary()
        self.attack_log_manager.flush()
        print()

    def attack(self, dataset, attack_args):
        """Attack `dataset` and record results to specified loggers.

        Args:
            dataset (textattack.datasets.Dataset): dataset to attack.
            attack_args (textattack.args.AttackArgs): arguments for attack.
        """
        assert isinstance(
            dataset, textattack.datasets.Dataset
        ), f"`dataset` argument must be of type `textattack.datasets.Dataset`, but got type of`{type(dataset)}`."
        assert isinstance(
            dataset, textattack.args.AttackArgs
        ), f"`attack_args` argument must be of type `textattack.args.AttackArgs`, but got type of `{type(attack_args)}`."

        textattack.shared.utils.set_seed(attack_args.random_seed)
        self._attack(dataset, attack_args=attack_args)

    def attack_parallel(self, dataset, attack_args, num_workers_per_device):
        """Attack `dataset` with single worker and record results to specified
        loggers.

        Args:
            dataset (textattack.datasets.Dataset): dataset to attack.
            num_workers_per_device (int): Number of worker threads to run per device. For example, if you are using GPUs and ``num_workers_per_device=2``,
                then 2 processes will be running in each GPU. If you are only using CPU, then this is equivalent to running 2 processes concurrently.
        """
        assert isinstance(
            dataset, textattack.datasets.Dataset
        ), f"`dataset` argument must be of type `textattack.datasets.Dataset`, but got type of `{type(dataset)}`."
        assert isinstance(
            dataset, textattack.args.AttackArgs
        ), f"`attack_args` argument must be of type `textattack.args.AttackArgs`, but got type of `{type(attack_args)}`."

        textattack.shared.utils.set_seed(attack_args.random_seed)
        self._attack_parallel(dataset, num_workers_per_device, attack_args=attack_args)

    def attack_interactive(self):
        print("Running in interactive mode")
        print("----------------------------")

        while True:
            print('Enter a sentence to attack or "q" to quit:')
            text = input()

            if text == "q":
                break

            if not text:
                continue

            print("Attacking...")

            example = textattack.shared.attacked_text.AttackedText(text)
            output = self.attack.goal_function.get_output(example)
            result = self.attack.attack(example, output)
            print(result.__str__(color_method="ansi") + "\n")

    def resume_attack(self, dataset, checkpoint):
        """Resume attacking `dataset` from saved `checkpoint`.

        Args:
            dataset (textattack.datasets.Dataset): dataset to attack.
            checkpoint (textattack.shared.AttackCheckpoint): checkpoint object that has previously been saved.
        """
        assert isinstance(
            dataset, textattack.datasets.Dataset
        ), f"`dataset` argument must be of type `textattack.datasets.Dataset`, but got type of `{type(dataset)}`."
        assert isinstance(
            dataset, textattack.shared.AttackCheckpoint
        ), f"`checkpoint` argument must be of type `textattack.shared.AttackCheckpoint`, but got type of `{type(checkpoint)}`."

        textattack.shared.utils.set_seed(checkpoint.attack_args.random_seed)
        self.attack_log_manager = checkpoint.attack_log_manager
        self._attack(dataset, checkpoint=checkpoint)

    def resume_attack_parallel(self, dataset, checkpoint, num_workers_per_device):
        """Resume attacking `dataset` from saved `checkpoint`.

        Args:
            dataset (textattack.datasets.Dataset): dataset to attack.
            checkpoint (textattack.shared.AttackCheckpoint): checkpoint object that has previously been saved.
        """
        assert isinstance(
            dataset, textattack.datasets.Dataset
        ), f"`dataset` argument must be of type `textattack.datasets.Dataset`, but got type of `{type(dataset)}`."
        assert isinstance(
            dataset, textattack.shared.AttackCheckpoint
        ), f"`checkpoint` argument must be of type `textattack.shared.AttackCheckpoint`, but got type of `{type(checkpoint)}`."

        textattack.shared.utils.set_seed(checkpoint.attack_args.random_seed)
        self.attack_log_manager = checkpoint.attack_log_manager
        self._attack(dataset, checkpoint=checkpoint)


def pytorch_multiprocessing_workaround():
    # This is a fix for a known bug
    try:
        torch.multiprocessing.set_start_method("spawn")
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass


def set_env_variables(gpu_id):
    # Disable tensorflow logs, except in the case of an error.
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Set sharing strategy to file_system to avoid file descriptor leaks
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Only use one GPU, if we have one.
    # For Tensorflow
    # TODO: Using USE with `--parallel` raises similar issue as https://github.com/tensorflow/tensorflow/issues/38518#
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # For PyTorch
    torch.cuda.set_device(gpu_id)

    # Fix TensorFlow GPU memory growth
    try:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                gpu = gpus[gpu_id]
                tf.config.experimental.set_visible_devices(gpu, "GPU")
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    except ModuleNotFoundError:
        pass


def attack_from_queue(attack_args, attack, in_queue, out_queue):
    gpu_id = torch.multiprocessing.current_process()._identity[0] - 2
    set_env_variables(gpu_id)
    textattack.shared.utils.set_seed(attack_args.random_seed)
    if gpu_id == 0:
        print(attack, "\n")
    while not in_queue.empty():
        try:
            i, example, ground_truth_output = in_queue.get()
            result = attack.attack(example, ground_truth_output)
            out_queue.put((i, result))
        except Exception as e:
            out_queue.put(e)
            exit()
