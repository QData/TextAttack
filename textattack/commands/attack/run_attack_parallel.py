"""A command line parser to run an attack from user specifications."""
from collections import deque
import os
import time

import torch
import tqdm

import textattack

from .attack_args_helpers import parse_attack_from_args, parse_dataset_from_args

logger = textattack.shared.logger


def set_env_variables(gpu_id):
    """Sets environment variables for a new worker on a given GPU."""
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


def attack_from_queue(args, in_queue, out_queue):
    """This function runs on each GPU and iteratively attacks samples from the
    in queue."""
    textattack.shared.logger.disabled = True
    textattack.shared.utils.set_seed(args.random_seed)
    gpu_id = torch.multiprocessing.current_process()._identity[0] - 2
    set_env_variables(gpu_id)
    attack = parse_attack_from_args(args)
    attack.loggers = []
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
    """The main thread.

    Starts a process on each GPU.
    """
    pytorch_multiprocessing_workaround()

    num_total_examples = args.num_examples

    if args.checkpoint_resume:
        args.num_remaining_attacks = checkpoint.num_remaining_attacks
        worklist = checkpoint.worklist
        worklist_tail = checkpoint.worklist_tail
        logger.info(
            "Recovered from checkpoint previously saved at {}".format(
                checkpoint.datetime
            )
        )
        print(checkpoint, "\n")
    else:
        args.num_remaining_attacks = num_total_examples
        worklist = deque(range(0, num_total_examples))
        worklist_tail = worklist[-1]

    # This makes `args` a namespace that's sharable between processes.
    # We could do the same thing with the model, but it's turned out
    # to be faster to let each thread have their own copy of the model.
    args = torch.multiprocessing.Manager().Namespace(**vars(args))

    num_gpus = torch.cuda.device_count()

    dataset = parse_dataset_from_args(args)

    textattack.shared.logger.info(f"Running on {num_gpus} GPUs")
    start_time = time.time()

    if args.interactive:
        raise RuntimeError("Cannot run in parallel if --interactive set")

    in_queue = torch.multiprocessing.Queue()
    out_queue = torch.multiprocessing.Queue()

    # Add stuff to queue.
    for i in worklist:
        try:
            text, output = dataset[i]
            in_queue.put((i, text, output))
        except IndexError:
            # if our dataset is shorter than the number of samples chosen, remove the
            # out-of-bounds indices from the worklist
            worklist.remove(i)

    # Store num_results, etc. in args.
    if args.checkpoint_resume:
        args.num_results = checkpoint.results_count
        args.num_failures = checkpoint.num_failed_attacks
        args.num_successes = checkpoint.num_successful_attacks
    else:
        args.num_results = 0
        args.num_failures = 0
        args.num_successes = 0
    # Start workers.
    pool = torch.multiprocessing.Pool(
        num_gpus, attack_from_queue, (args, in_queue, out_queue)
    )
    pool.close()

    # Disable CUDA on the main worker. Use this attack specifically for logging
    # purposes.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # import tensorflow as tf
    # tf.config.experimental.set_visible_devices([], 'GPU')
    textattack.shared.device = torch.device("cpu")
    attack = parse_attack_from_args(args)
    del attack.constraints
    del attack.goal_function
    del attack.search_method
    del attack.transformation

    # Log results asynchronously and update progress bar.
    num_results = args.num_results
    num_failures = args.num_failures
    num_successes = args.num_successes

    pbar = tqdm.tqdm(total=args.num_remaining_attacks, smoothing=0)
    while worklist:
        result = out_queue.get(block=True)

        if isinstance(result, Exception):
            raise result
        idx, result = result
        # Set result label names.
        result.original_result.attacked_text.attack_attrs[
            "label_names"
        ] = dataset.label_names
        result.perturbed_result.attacked_text.attack_attrs[
            "label_names"
        ] = dataset.label_names

        attack.log_result(result)
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
            and len(attack.results) % args.checkpoint_interval == 0
        ):
            new_checkpoint = textattack.shared.Checkpoint(
                args, attack, worklist, worklist_tail
            )
            new_checkpoint.save()

    pbar.close()
    print()
    # Enable summary stdout.
    if args.disable_stdout:
        attack.enable_stdout()
    attack.log_metrics()

    print()

    textattack.shared.logger.info(f"Attack time: {time.time() - start_time}s")

    return attack.attack_results


def pytorch_multiprocessing_workaround():
    # This is a fix for a known bug. See https://github.com/pytorch/pytorch/issues/11201
    try:
        torch.multiprocessing.set_start_method("spawn")
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
