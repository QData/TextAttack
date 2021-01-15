import collections
import logging
import multiprocessing as mp
import os

import dill
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
from .attack_args import AttackArgs


class Attacker:
    """Class for running attacks on a dataset with specified parameters. This
    class uses the ``textattack.Attack`` to actually run the attacks, while
    also providing useful features such as parallel processing, saving/resuming
    from a checkpint, logging to files and stdout.

    Args:
        attack_or_attack_build_fn (Union[textattack.Attack, callable]): ``Attack`` object or a zero-argument callable that returns the desired ``Attack`` object.
            Zero-argument callable is required when attacking in parallel because the attack has to be newly initialized in each worker process (it is slow and difficult to
            seralize ``Attack`` and share it across proccesses).
        dataset (textattack.datasets.Dataset): Dataset to attack.
        attack_args (textattack.AttackArgs): Arguments for attacking the dataset. For default settings, look at the `AttackArgs` class.

    Examples::

        >>> import textattack
        >>> import transformers

        >>> model = transformers.AutoModelForSequenceClassification("textattack/bert-base-uncased-imdb")
        >>> tokenizer = transformers.AutoTokenizer("textattack/bert-base-uncased-imdb")
        >>> model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

        >>> attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
        >>> dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

        >>> # Attack 1000 samples with CSV logging and checkpoint saved every 100 interval
        >>> attack_args = textattack.AttackArgs(num_examples=1000, log_to_csv="log.csv", checkpoint_interval=100, checkpoint_dir="checkpoints")

        >>> attacker = textattack.Attacker(attack, dataset, attack_args)
        >>> attacker.attack_dataset()
    """

    def __init__(self, attack_or_attack_build_fn, dataset, attack_args=AttackArgs()):
        assert isinstance(attack_or_attack_build_fn, Attack) or callable(
            attack_or_attack_build_fn
        ), f"`attack` argument must be of type `textattack.Attack` or a callable, but got type of `{type(attack_or_attack_build_fn)}`."
        assert isinstance(
            dataset, textattack.datasets.Dataset
        ), f"`dataset` must be of type `textattack.datasets.Dataset`, but got type `{type(dataset)}`."
        assert isinstance(
            attack_args, textattack.AttackArgs
        ), f"`attack_args` must be of type `textattack.AttackArgs`, but got type `{type(attack_args)}`."

        if isinstance(attack_or_attack_build_fn, Attack):
            if attack_args.parallel:
                raise ValueError(
                    "To performing parallel attacks, `attack_or_attack_build_fn` must "
                    "be a zero-argument callable that returns the desired `Attack` object."
                )
            else:
                self.attack = attack_or_attack_build_fn
        else:
            self.attack = None
            self._attack_build_fn = attack_or_attack_build_fn

        self.dataset = dataset
        self.attack_args = attack_args
        self.attack_log_manager = AttackArgs.create_loggers_from_args(attack_args)

        # This is to be set if loading from a checkpoint
        self._checkpoint = None

    def update_attack_args(self, **kwargs):
        """To update any attack args, pass the new argument as keyword argument
        to this function.

        Examples::

        >>> attacker = #some instance of Attacker
        >>> # To switch to parallel mode and increase checkpoint interval from 100 to 500
        >>> attacker.update_attack_args(parallel=True, checkpoint_interval=500)
        """
        for k in kwargs:
            if hasattr(self.attack_args, k):
                self.attack_args.k = kwargs[k]
            else:
                raise ValueError(f"`AttackArgs` does not have field {k}.")

    def attack_dataset(self):
        """Start the attack."""
        textattack.shared.utils.set_seed(self.attack_args.random_seed)
        if self.dataset.shuffled and self.attack_args.checkpoint_interval:
            # Not allowed b/c we cannot recover order of shuffled data
            raise ValueError("Cannot use `--checkpoint-interval` with `--shuffle=True`")

        self.attack_args.num_examples = (
            len(self.dataset)
            if self.attack_args.num_examples == -1
            else self.attack_args.num_examples
        )
        if self.attack_args.parallel:
            if torch.cuda.device_count() == 0:
                raise Exception(
                    "Found no GPU on your system. To run attacks in parallel, GPU is required."
                )
            self._attack_parallel()
        else:
            if self.attack is None:
                self.attack = self._attack_build_fn()
                assert isinstance(
                    self.attack, Attack
                ), f"`attack_or_attack_build_fn` must return an instance of `Attack` class, but return type `{type(self.attack)}`."
            self._attack()

    def _attack(self):
        """Internal method that carries out attack.

        No parallel processing is involved.
        """
        if self._checkpoint:
            num_remaining_attacks = self._checkpoint.num_remaining_attacks
            worklist = self._checkpoint.worklist
            worklist_tail = self._checkpoint.worklist_tail
            logger.info(
                f"Recovered from checkpoint previously saved at {self._checkpoint.datetime}."
            )
        else:
            num_remaining_attacks = self.attack_args.num_examples
            start = self.attack_args.num_examples_offset
            end = start + self.attack_args.num_examples
            worklist = collections.deque(range(start, end))
            worklist_tail = worklist[-1]

        print(self.attack, "\n")

        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0)
        if self._checkpoint:
            num_results = self._checkpoint.results_count
            num_failures = self._checkpoint.num_failed_attacks
            num_successes = self._checkpoint.num_successful_attacks
        else:
            num_results = 0
            num_failures = 0
            num_successes = 0

        i = 0
        while worklist:
            idx = worklist.popleft()
            i += 1
            example, ground_truth_output = self.dataset[idx]
            example = textattack.shared.AttackedText(example)
            if self.dataset.label_names is not None:
                example.attack_attrs["label_names"] = self.dataset.label_names
            result = self.attack.attack(example, ground_truth_output)
            self.attack_log_manager.log_result(result)

            if not self.attack_args.disable_stdout:
                print("\n")

            if isinstance(result, SkippedAttackResult) and self.attack_args.attack_n:
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
                self.attack_args.checkpoint_interval
                and len(self.attack_log_manager.results)
                % self.attack_args.checkpoint_interval
                == 0
            ):
                new_checkpoint = textattack.shared.AttackCheckpoint(
                    self.attack_args, self.attack_log_manager, worklist, worklist_tail
                )
                new_checkpoint.save()
                self.attack_log_manager.flush()

        pbar.close()
        print()
        # Enable summary stdout
        if self.attack_args.disable_stdout:
            self.attack_log_manager.enable_stdout()
        self.attack_log_manager.log_summary()
        self.attack_log_manager.flush()
        print()

    def _attack_parallel(self):
        pytorch_multiprocessing_workaround()

        if self._checkpoint:
            num_remaining_attacks = self._checkpoint.num_remaining_attacks
            worklist = self._checkpoint.worklist
            worklist_tail = self._checkpoint.worklist_tail
            logger.info(
                f"Recovered from checkpoint previously saved at {self._checkpoint.datetime}."
            )
        else:
            num_remaining_attacks = self.attack_args.num_examples
            start = self.attack_args.num_examples_offset
            end = start + self.attack_args.num_examples
            worklist = collections.deque(range(start, end))
            worklist_tail = worklist[-1]

        in_queue = torch.multiprocessing.Queue()
        out_queue = torch.multiprocessing.Queue()
        for i in worklist:
            try:
                example, ground_truth_output = self.dataset[i]
                example = textattack.shared.AttackedText(example)
                if self.dataset.label_names is not None:
                    example.attack_attrs["label_names"] = self.dataset.label_names
                in_queue.put((i, example, ground_truth_output))
            except IndexError:
                raise IndexError(
                    f"Tried to access element at {i} in dataset of size {len(self.dataset)}."
                )

        # We reserve the first GPU for coordinating workers.
        num_gpus = torch.cuda.device_count()
        num_workers = self.attack_args.num_workers_per_device * num_gpus
        logger.info(f"Running {num_workers} worker(s) on {num_gpus} GPU(s).")

        # Barrier for synchronizing worker processes.
        # We want them to all wait until last process has loaded its attack
        barrier = mp.Barrier(num_workers)

        # Start workers.
        # Use dill to serialize the function/callable.
        attack_build_fn_str = dill.dumps(self._attack_build_fn)
        torch.multiprocessing.Pool(
            num_workers,
            attack_from_queue,
            (
                self.attack_args,
                attack_build_fn_str,
                num_gpus,
                barrier,
                in_queue,
                out_queue,
            ),
        )
        # Log results asynchronously and update progress bar.
        if self._checkpoint:
            num_results = self._checkpoint.results_count
            num_failures = self._checkpoint.num_failed_attacks
            num_successes = self._checkpoint.num_successful_attacks
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
            if self.attack_args.attack_n and isinstance(result, SkippedAttackResult):
                # worklist_tail keeps track of highest idx that has been part of worklist
                # Used to get the next dataset element when attacking with `attack_n` = True.
                worklist_tail += 1
                try:
                    example, ground_truth_output = self.dataset[worklist_tail]
                    example = textattack.shared.AttackedText(example)
                    if self.dataset.label_names is not None:
                        example.attack_attrs["label_names"] = self.dataset.label_names
                    worklist.append(worklist_tail)
                    in_queue.put((worklist_tail, example, ground_truth_output))
                except IndexError:
                    logger.warn(
                        f"Attempted to attack {self.attack_args.num_examples} examples with but ran out of examples. "
                        f"You might see fewer number of results than {self.attack_args.num_examples}."
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
                self.attack_args.checkpoint_interval
                and len(self.attack_log_manager.results)
                % self.attack_args.checkpoint_interval
                == 0
            ):
                new_checkpoint = textattack.shared.AttackCheckpoint(
                    self.attack_args, self.attack_log_manager, worklist, worklist_tail
                )
                new_checkpoint.save()
                self.attack_log_manager.flush()

        pbar.close()
        print()
        # Enable summary stdout.
        if self.attack_args.disable_stdout:
            self.attack_log_manager.enable_stdout()
        self.attack_log_manager.log_summary()
        self.attack_log_manager.flush()
        print()

    @classmethod
    def from_checkpoint(cls, attack_or_attack_build_fn, dataset, checkpoint):
        """Resume attacking from a saved checkpoint. Attacker and dataset must
        be recovered by the user again, while attack args are loaded from the
        saved checkpoint.

        Args:
            attack_or_attack_build_fn (Union[textattack.Attack, callable]): ``Attack`` object or a zero-argument callable that returns the desired ``Attack`` object.
                Zero-argument callable is required when attacking in parallel because the attack has to be newly initialized in each worker process (it is slow and difficult to
                seralize ``Attack`` and share it across proccesses).
            dataset (textattack.datasets.Dataset): Dataset to attack.
            checkpoint (Union[str, textattack.shared.AttackChecpoint]): Saved checkpoint or path of the saved checkpoint.
        """
        assert isinstance(
            checkpoint, (str, textattack.shared.AttackCheckpoint)
        ), f"`checkpoint` must be of type `str` or `textattack.shared.AttackCheckpoint`, but got type `{type(checkpoint)}`."

        if isinstance(checkpoint, str):
            checkpoint = textattack.shared.AttackCheckpoint.load(checkpoint)
        attacker = cls(attack_or_attack_build_fn, dataset, checkpoint.attack_args)
        attacker.attack_log_manager = checkpoint.attack_log_manager
        attacker._checkpoint = checkpoint
        return attacker

    @staticmethod
    def attack_interactive(attack):
        print(attack, "\n")

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
            output = attack.goal_function.get_output(example)
            result = attack.attack(example, output)
            print(result.__str__(color_method="ansi") + "\n")


#
# Helper Methods for multiprocess attacks
#
def pytorch_multiprocessing_workaround():
    # This is a fix for a known bug
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
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


def attack_from_queue(
    attack_args, attack_build_fn_str, num_gpus, barrier, in_queue, out_queue
):
    gpu_id = (torch.multiprocessing.current_process()._identity[0] - 1) % num_gpus
    set_env_variables(gpu_id)
    textattack.shared.utils.set_seed(attack_args.random_seed)
    if torch.multiprocessing.current_process()._identity[0] > 1:
        logging.disable()

    attack_build_fn = dill.loads(attack_build_fn_str)
    attack = attack_build_fn()
    assert isinstance(
        attack, Attack
    ), f"`attack_build_fn` must return an instance of `Attack` class, but return type `{type(attack)}`."

    proc = barrier.wait()
    if proc == 0:
        # Print attack if it's the last process to call `barrier.wait()`
        print(attack, "\n")

    while not in_queue.empty():
        try:
            i, example, ground_truth_output = in_queue.get()
            result = attack.attack(example, ground_truth_output)
            out_queue.put((i, result))
        except Exception as e:
            out_queue.put(e)
            exit()
