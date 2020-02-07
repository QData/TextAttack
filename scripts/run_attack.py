"""
A command line parser to run an attack from user specifications.
"""

import textattack
import time
import tqdm
import os

from textattack_args_helper import *

def main():
    args = get_args()
    
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
    model, attack = parse_model_and_attack_from_args(args)
    
    # Logger
    attack_logger = parse_logger_from_args(args)

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

            tokenized_text = textattack.shared.tokenized_text.TokenizedText(text, model.tokenizer)
            
            pred = attack._call_model([tokenized_text])
            label = int(pred.argmax())

            print('Attacking...')

            result = next(attack.attack_dataset([(label, text)]))
            print(result.__str__(color_method='stdout'))
    
    else:
        # Not interactive? Use default dataset.
        if args.model in DATASET_BY_MODEL:
            data = DATASET_BY_MODEL[args.model](offset=args.num_examples_offset)
        else:
            raise ValueError(f'Error: unsupported model {args.model}')
        
        pbar = tqdm.tqdm(total=args.num_examples)
        for result in attack.attack_dataset(data, 
            num_examples=args.num_examples, shuffle=args.shuffle):
            attack_logger.log_result(result)
            if not args.disable_stdout:
                print('\n')
            else:
                pbar.update(1)
        pbar.close()
        print()
        # Enable summary stdout
        if args.disable_stdout:
            attack_logger.enable_stdout()
        attack_logger.log_summary()
        attack_logger.flush()
        print()
        finish_time = time.time()
        print(f'Attack time: {time.time() - load_time}s')

if __name__ == '__main__': main()