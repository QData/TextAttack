"""
A command line parser to run an attack from user specifications.
"""

import os
import textattack
import time
import torch
import tqdm

from .run_attack_args_helper import *

def set_env_variables(gpu_id):
    # Set sharing strategy to file_system to avoid file descriptor leaks
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Only use one GPU, if we have one.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # Disable tensorflow logs, except in the case of an error.
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Cache TensorFlow Hub models here, if not otherwise specified.
    if 'TFHUB_CACHE_DIR' not in os.environ:
        os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser('~/.cache/tensorflow-hub')

def attack_from_queue(args, in_queue, out_queue):
    gpu_id = torch.multiprocessing.current_process()._identity[0] - 2
    set_env_variables(gpu_id)
    _, attack = parse_goal_function_and_attack_from_args(args)
    if gpu_id == 0:
        print(attack, '\n')
    while not in_queue.empty():
        try:
            output, text = in_queue.get()
            results_gen = attack.attack_dataset([(output, text)], num_examples=1)
            result = next(results_gen)
            out_queue.put(result)
        except Exception as e:
            out_queue.put(e)
            exit()

def run(args):
    pytorch_multiprocessing_workaround()

    if args.checkpoint_resume:
        # Override current args with checkpoint args
        resume_checkpoint = parse_checkpoint_from_args(args)
        args = merge_checkpoint_args(resume_checkpoint.args, args)
        num_examples_offset = resume_checkpoint.dataset_offset
        num_examples = resume_checkpoint.num_remaining_attack
        logger.info('Recovered from previously saved checkpoint at {}'.format(resume_checkpoint.datetime))
        print(resume_checkpoint, '\n')
    else:
        num_examples_offset = args.num_examples_offset
        num_examples = args.num_examples

    # This makes `args` a namespace that's sharable between processes.
    # We could do the same thing with the model, but it's actually faster
    # to let each thread have their own copy of the model.
    args = torch.multiprocessing.Manager().Namespace(
        **vars(args)
    )
    start_time = time.time()
    
    if args.checkpoint_resume:
        attack_log_manager = resume_checkpoint.log_manager
    else:
        attack_log_manager = parse_logger_from_args(args)
    
    # We reserve the first GPU for coordinating workers.
    num_gpus = torch.cuda.device_count()
    dataset = DATASET_BY_MODEL[args.model](offset=num_examples_offset)
    
    print(f'Running on {num_gpus} GPUs')
    load_time = time.time()

    if args.interactive:
        raise RuntimeError('Cannot run in parallel if --interactive set')
    
    in_queue = torch.multiprocessing.Queue()
    out_queue =  torch.multiprocessing.Queue()
    # Add stuff to queue.
    for _ in range(num_examples):
        label, text = next(dataset)
        in_queue.put((label, text))
    # Start workers.
    pool = torch.multiprocessing.Pool(
        num_gpus, 
        attack_from_queue, 
        (args, in_queue, out_queue)
    )
    # Log results asynchronously and update progress bar.
    if args.checkpoint_resume:
        num_results = resume_checkpoint.results_count
        num_failures = resume_checkpoint.num_failed_attacks
        num_successes = resume_checkpoint.num_successful_attacks
    else:
        num_results = 0
        num_failures = 0
        num_successes = 0
    pbar = tqdm.tqdm(total=num_examples, smoothing=0)
    while num_results < num_examples:
        result = out_queue.get(block=True)
        if isinstance(result, Exception):
            raise result
        attack_log_manager.log_result(result)
        if (not args.attack_n) or (not isinstance(result, textattack.attack_results.SkippedAttackResult)):
            pbar.update()
            num_results += 1
            if type(result) == textattack.attack_results.SuccessfulAttackResult:
                num_successes += 1
            if type(result) == textattack.attack_results.FailedAttackResult:
                num_failures += 1
            pbar.set_description('[Succeeded / Failed / Total] {} / {} / {}'.format(num_successes, num_failures, num_results))
        else:
            label, text = next(dataset)
            in_queue.put((label, text))

        if args.checkpoint_interval and num_results % args.checkpoint_interval == 0:
            checkpoint = textattack.shared.Checkpoint(chkpt_time, args, attack_log_manager)
            checkpoint.save()
            attack_log_manager.flush()

    pbar.close()
    print()
    # Enable summary stdout.
    if args.disable_stdout:
        attack_log_manager.enable_stdout()
    attack_log_manager.log_summary()
    attack_log_manager.flush()
    print()
    finish_time = time.time()
    print(f'Attack time: {time.time() - load_time}s')

def pytorch_multiprocessing_workaround():
    # This is a fix for a known bug
    try:
        torch.multiprocessing.set_start_method('spawn')
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass

if __name__ == '__main__': 
    run(get_args())
