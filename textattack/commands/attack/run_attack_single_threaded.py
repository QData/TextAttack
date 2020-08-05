"""A command line parser to run an attack from user specifications."""
from collections import deque
import os
import time

import tqdm

import textattack

from .attack_args_helpers import (
    parse_attack_from_args,
    parse_dataset_from_args,
    parse_logger_from_args,
)

logger = textattack.shared.logger


def run(args, checkpoint=None):
    # Only use one GPU, if we have one.
    # TODO: Running Universal Sentence Encoder uses multiple GPUs
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Disable tensorflow logs, except in the case of an error.
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # Fix TensorFlow GPU memory growth
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

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
        num_remaining_attacks = args.num_examples
        worklist = deque(range(0, args.num_examples))
        worklist_tail = worklist[-1]

    start_time = time.time()

    # Attack
    attack = parse_attack_from_args(args)
    print(attack, "\n")

    # Logger
    if args.checkpoint_resume:
        attack_log_manager = checkpoint.log_manager
    else:
        attack_log_manager = parse_logger_from_args(args)

    load_time = time.time()
    textattack.shared.logger.info(f"Load time: {load_time - start_time}s")

    if args.interactive:
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

            attacked_text = textattack.shared.attacked_text.AttackedText(text)
            initial_result = attack.goal_function.get_output(attacked_text)
            result = next(attack.attack_dataset([(text, initial_result)]))
            print(result.__str__(color_method="ansi") + "\n")

    else:
        # Not interactive? Use default dataset.
        dataset = parse_dataset_from_args(args)

        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0)
        if args.checkpoint_resume:
            num_results = checkpoint.results_count
            num_failures = checkpoint.num_failed_attacks
            num_successes = checkpoint.num_successful_attacks
        else:
            num_results = 0
            num_failures = 0
            num_successes = 0

        for result in attack.attack_dataset(dataset, indices=worklist):
            attack_log_manager.log_result(result)

            if not args.disable_stdout:
                print("\n")
            if (not args.attack_n) or (
                not isinstance(result, textattack.attack_results.SkippedAttackResult)
            ):
                pbar.update(1)
            else:
                # worklist_tail keeps track of highest idx that has been part of worklist
                # Used to get the next dataset element when attacking with `attack_n` = True.
                worklist_tail += 1
                worklist.append(worklist_tail)

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
        # Enable summary stdout
        if args.disable_stdout:
            attack_log_manager.enable_stdout()
        attack_log_manager.log_summary()
        attack_log_manager.flush()
        print()
        # finish_time = time.time()
        textattack.shared.logger.info(f"Attack time: {time.time() - load_time}s")

        return attack_log_manager.results
