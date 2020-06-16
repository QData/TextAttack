"""
A command line parser to run an attack from user specifications.
"""

import textattack
import time
import tqdm
import os
import datetime
from collections import deque

from .attack_args_helper import *

logger = textattack.shared.logger

def run(args):
    # Only use one GPU, if we have one.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Disable tensorflow logs, except in the case of an error.
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Cache TensorFlow Hub models here, if not otherwise specified.
    if 'TFHUB_CACHE_DIR' not in os.environ:
        os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser('~/.cache/tensorflow-hub')

    if args.checkpoint_resume:
        # Override current args with checkpoint args
        resume_checkpoint = parse_checkpoint_from_args(args)
        args = merge_checkpoint_args(resume_checkpoint.args, args)
        
        num_remaining_attacks = resume_checkpoint.num_remaining_attacks
        worklist = resume_checkpoint.worklist.copy()
        assert num_remaining_attacks == len(worklist), "Recorded number of remaining attacks and size of worklist are different."
        last_example = resume_checkpoint.last_example

        logger.info('Recovered from checkpoint previously saved at {}'.format(resume_checkpoint.datetime))
        print(resume_checkpoint, '\n')
    else:
        num_remaining_attacks = args.num_examples
        worklist = deque(range(0, args.num_examples))
        last_example = worklist[-1]
    
    start_time = time.time()
    
    # Attack
    attack = parse_attack_from_args(args)
    print(attack, '\n')
    
    # Logger
    if args.checkpoint_resume:
        attack_log_manager = resume_checkpoint.log_manager
    else:
        attack_log_manager = parse_logger_from_args(args)

    load_time = time.time()
    print(f'Load time: {load_time - start_time}s')

    if args.interactive:
        print('Running in interactive mode')
        print('----------------------------')

        while True:
            print('Enter a sentence to attack or "q" to quit:')
            text = input()

            if text == 'q':
                break
            
            if not text:
                continue

            print('Attacking...')
            
            tokenized_text = textattack.shared.tokenized_text.TokenizedText(text, attack.goal_function.model.tokenizer)
            initial_result = attack.goal_function.get_output(tokenized_text)
            result = next(attack.attack_dataset([(text, initial_result)]))
            print(result.__str__(color_method='ansi'))
    
    else:
        # Not interactive? Use default dataset.
        dataset = parse_dataset_from_args(args)
        
        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0)
        if args.checkpoint_resume:
            num_results = resume_checkpoint.results_count
            num_failures = resume_checkpoint.num_failed_attacks
            num_successes = resume_checkpoint.num_successful_attacks
        else:
            num_results = 0
            num_failures = 0
            num_successes = 0
        i = 0
        for result in attack.attack_dataset(dataset, 
                                        indicies=worklist,
                                        attack_n=args.attack_n):
            attack_log_manager.log_result(result)

            if not args.disable_stdout:
                print('\n')
            if (not args.attack_n) or (not isinstance(result, textattack.attack_results.SkippedAttackResult)):
                pbar.update(1)
            else:
                last_example += 1
                worklist.append(last_example)
            num_results += 1
            try:
                worklist.popleft()
                i += 1
            except ValueError:
                raise Exception('{} does not exist in the worklist.'.format(worklist[i]))
            if type(result) == textattack.attack_results.SuccessfulAttackResult:
                num_successes += 1
            if type(result) == textattack.attack_results.FailedAttackResult:
                num_failures += 1
            pbar.set_description('[Succeeded / Failed / Total] {} / {} / {}'.format(num_successes, num_failures, num_results))

            if args.checkpoint_interval and num_results % args.checkpoint_interval == 0:
                checkpoint = textattack.shared.Checkpoint(args, attack_log_manager, worklist, last_example)
                if not checkpoint.verify_no_duplicates():
                    logger.warning('Duplicate results processed.')
                checkpoint.save()
                attack_log_manager.flush()

        pbar.close()
        print()
        # Enable summary stdout
        if args.disable_stdout:
            attack_log_manager.enable_stdout()
        attack_log_manager.log_summary()
        attack_log_manager.flush()
        print()
        finish_time = time.time()
        print(f'Attack time: {time.time() - load_time}s')

if __name__ == '__main__': 
    run(get_args())
