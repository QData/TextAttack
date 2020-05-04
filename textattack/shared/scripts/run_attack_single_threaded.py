"""
A command line parser to run an attack from user specifications.
"""

import textattack
import time
import tqdm
import os

from .run_attack_args_helper import *

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
    
    start_time = time.time()
    
    # Models and Attack
    goal_function, attack = parse_goal_function_and_attack_from_args(args)
    print(attack, '\n')
    
    # Logger
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

            tokenized_text = textattack.shared.tokenized_text.TokenizedText(text, goal_function.model.tokenizer)
            
            result = goal_function.get_result(tokenized_text, goal_function.get_output(tokenized_text))
            print('Attacking...')

            result = next(attack.attack_dataset([(text, result.output)]))
            print(result.__str__(color_method='stdout'))
    
    else:
        # Not interactive? Use default dataset.
        if args.model in DATASET_BY_MODEL:
            data = DATASET_BY_MODEL[args.model](offset=args.num_examples_offset)
        else:
            raise ValueError(f'Error: unsupported model {args.model}')
        
        pbar = tqdm.tqdm(total=args.num_examples, smoothing=0)
        num_results = 0
        num_failures = 0
        num_successes = 0
        for result in attack.attack_dataset(data, 
                                        num_examples=args.num_examples, 
                                        shuffle=args.shuffle, 
                                        attack_n=args.attack_n):
            attack_log_manager.log_result(result)
            if not args.disable_stdout:
                print('\n')
            if (not args.attack_n) or (not isinstance(result, textattack.attack_results.SkippedAttackResult)):
                pbar.update(1)
            num_results += 1
            if type(result) == textattack.attack_results.SuccessfulAttackResult:
                num_successes += 1
            if type(result) == textattack.attack_results.FailedAttackResult:
                num_failures += 1
            pbar.set_description('[Succeeded / Failed / Total] {} / {} / {}'.format(num_successes, num_failures, num_results))
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
