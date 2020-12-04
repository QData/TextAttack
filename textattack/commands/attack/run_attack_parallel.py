"""

TextAttack Command Class for Attack Parralle
---------------------------------------------

A command line parser to run an attack in parralle from user specifications.

"""


from collections import deque
import os
import time

import torch
import tqdm

import textattack

from .attack_args_helpers import (
    parse_attack_from_args,
    parse_dataset_from_args,
    parse_logger_from_args,
)

logger = textattack.shared.logger


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


def attack_from_queue(args, in_queue, out_queue):
    gpu_id = torch.multiprocessing.current_process()._identity[0] - 2
    set_env_variables(gpu_id)
    textattack.shared.utils.set_seed(args.random_seed)
    attack = parse_attack_from_args(args)
    if gpu_id == 0:
        print(attack, "\n")
    while not in_queue.empty():
        try:
            i, text, output = in_queue.get()
            results_gen = attack.attack_dataset([(text, output)])
            result = next(results_gen)
            out_queue.put((i, result))
        except Exception as e:
            out_queue.put(e)
            exit()


def run(args, checkpoint=None):
    pytorch_multiprocessing_workaround()

    dataset = parse_dataset_from_args(args)
    num_total_examples = args.num_examples

    if args.checkpoint_resume:
        num_remaining_attacks = checkpoint.num_remaining_attacks
        worklist = checkpoint.worklist
        worklist_tail = checkpoint.worklist_tail
        logger.info(
            "Recovered from checkpoint previously saved at {}".format(
                checkpoint.datetime
            )
        )
        print(checkpoint, "\n")
    else:
        num_remaining_attacks = num_total_examples
        worklist = deque(range(0, num_total_examples))
        worklist_tail = worklist[-1]

    # This makes `args` a namespace that's sharable between processes.
    # We could do the same thing with the model, but it's actually faster
    # to let each thread have their own copy of the model.
    args = torch.multiprocessing.Manager().Namespace(**vars(args))

    if args.checkpoint_resume:
        attack_log_manager = checkpoint.log_manager
    else:
        attack_log_manager = parse_logger_from_args(args)

    # We reserve the first GPU for coordinating workers.
    num_gpus = torch.cuda.device_count()

    textattack.shared.logger.info(f"Running on {num_gpus} GPUs")
    start_time = time.time()

    if args.interactive:
        raise RuntimeError("Cannot run in parallel if --interactive set")

    in_queue = torch.multiprocessing.Queue()
    out_queue = torch.multiprocessing.Queue()
    # Add stuff to queue.
    missing_datapoints = set()
    for i in worklist:
        try:
            text, output = dataset[i]
            in_queue.put((i, text, output))
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

    textattack.shared.logger.info(f"Attack time: {time.time() - start_time}s")

    return attack_log_manager.results


def pytorch_multiprocessing_workaround():
    # This is a fix for a known bug
    try:
        torch.multiprocessing.set_start_method("spawn")
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
