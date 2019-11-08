'''
A command line parser to run an attack

'''

import argparse
import time
import os
import inspect
import re
import ast

import textattack.datasets as datasets
import textattack.attacks as attacks
import textattack.models as models
import textattack.constraints as constraints
import textattack.transformations as transformations

from textattack.tokenized_text import TokenizedText


DATASET_CLASS_NAMES = {
    'agnews':           datasets.classification.AGNews,
    'imdb':             datasets.classification.IMDBSentiment,
    'kaggle_fake_news': datasets.classification.KaggleFakeNews,
    'mr':               datasets.classification.MovieReviewSentiment,
    'yelp':             datasets.classification.YelpSentiment
}


def get_args():
    parser = argparse.ArgumentParser(description='A commandline parser for TextAttack', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    data_group = parser.add_mutually_exclusive_group(required=True)
    
    parser.add_argument('--attack', type=str, required=False, default='greedy-wir-counterfit', 
        help='The type of attack to run.')
    
    parser.add_argument('--model', type=str, required=False, default='bert-sentiment',
        help='The classification model to attack.')

    parser.add_argument('--transformations', type=str, required=False, nargs='*',
        default=['word-swap-embedding'], help='Transformation(s) to add to the attack')
    
    parser.add_argument('--constraints', type=str, required=False, nargs='*',
        default=['use'], 
        help=('Constraints to add to the attack. Usage: "--constraints {constraint}:{arg_1}={value_1},{arg_3}={value_3}' 
        ' Options are use, lang-tool, and goog-lm'))
    
    parser.add_argument('--out_file', type=str, required=False,
        help='The file to output the results to.')
    
    parser.add_argument('--num_examples', type=int, required=False, 
        help='The number of examples to attack.')
    
    data_group.add_argument('--interactive', action='store_true', 
        help='Whether to run attacks interactively.')
    
    data_group.add_argument('--data', type=str,
        choices=DATASET_CLASS_NAMES.keys(), help='The dataset to use.')
    
    args = parser.parse_args()
    
    return args


def get_params(func_callable, function_params):

    # Break the string passed into argparse into the function and each parameter
    param_list = re.split(':|,', function_params)

    sig = inspect.signature(func_callable)

    # Make a dictionary mapping each parameter to its default value or None
    args = dict(str(item).split("=") if len(str(item).split("=")) == 2 else [str(item), None] for item in sig.parameters.values())

    for param in param_list[1:]:
        keyword, value = param.split('=')

        if keyword not in args.keys():
            raise ValueError(f'{keyword} is not a valid parameter for {param_list[0]}')
        else:
            args[keyword] = value

    return args


if __name__ == '__main__':
    args = get_args()
    
    # Only use one GPU, if we have one.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Disable tensorflow logs, except in the case of an error.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    start_time = time.time()

    # Models
    if 'bert-sentiment' in args.model:
        params = get_params(models.BertForSentimentClassification, args.model)
        model = models.BertForSentimentClassification(max_seq_length=float(params['max_seq_length']))

    else:
        raise ValueError('Model must be bert-sentiment')
    
    # Transformations
    if args.transformations:

        added_transformations = []

        for transformation in args.transformations:

            if 'word-swap-embedding' in transformation:
                params = get_params(transformations.WordSwapEmbedding, transformation)
                added_transformations.append(transformations.WordSwapEmbedding(
                    replace_stopwords = ast.literal_eval(params['replace_stopwords']),
                    word_embedding = params['word_embedding'].replace("'", ''),
                    similarity_threshold = ast.literal_eval(params['similarity_threshold'])
                ))

            else:
                raise ValueError((f'{transformation} is not a valid transformation type. '
                'Valid transformations are word-swap-embedding'))

    # Attacks
    if 'greedy-counterfit' in args.attack:
        params = get_params(attacks.GreedyWordSwap, args.attack)
        attack = attacks.GreedyWordSwap(
            model, 
            added_transformations, 
            max_depth=float(params['max_depth']
        ))

    elif 'ga-counterfit' in args.attack:
        params = get_params(attacks.GeneticAlgorithm, args.attack)
        attack = attacks.GeneticAlgorithm(
            model, 
            added_transformations, 
            pop_size=int(params['pop_size']),
            max_iters=int(params['max_iters'])
            )

    elif 'greedy-wir-counterfit' in args.attack:
        params = get_params(attacks.GreedyWordSwapWIR, args.attack)
        attack = attacks.GreedyWordSwapWIR(
            model, 
            added_transformations, 
            max_depth=float(params['max_depth']
        ))

    else:
        raise ValueError(f'{args.attack} is not a valid attack')

    # Constraints
    if args.constraints:

        defined_constraints = []

        for constraint in args.constraints:

            if 'use' in constraint:
                params = get_params(constraints.semantics.UniversalSentenceEncoder, constraint)
                defined_constraints.append(constraints.semantics.UniversalSentenceEncoder(
                    threshold=float(params['threshold']), 
                    metric=params['metric'].replace("'", '')
                    ))

            elif 'lang-tool' in constraint:
                params = get_params(constraints.syntax.LanguageTool, constraint)
                defined_constraints.append(constraints.syntax.LanguageTool(
                    threshold=float(params['threshold'])
                    ))

            elif 'goog-lm' in constraint:
                params = get_params(constraints.semantics.google_language_model.GoogleLanguageModel, constraint)
                defined_constraints.append(constraints.semantics.google_language_model.GoogleLanguageModel(
                    top_n = ast.literal_eval(params['top_n']),
                    top_n_per_index = ast.literal_eval(params['top_n_per_index']),
                    print_step = ast.literal_eval(params['print_step'])
                ))

            else:
                raise ValueError((f'{constraint} is not a valid constraint. ' 
                    'Valid options are "use", "lang-tool", or "goog-lm". Use "-h" for help.'))

        attack.add_constraints(defined_constraints)

    # Data
    if args.data in DATASET_CLASS_NAMES:
        dataset_class = DATASET_CLASS_NAMES[args.data]
        data = dataset_class(args.num_examples)

    # Output file
    if args.out_file is not None:
        attack.add_output_file(args.out_file)


    load_time = time.time()

    if args.data is not None:
        attack.attack(data, shuffle=False)

        finish_time = time.time()

        print(f'Loaded in {load_time - start_time}s')
        print(f'Ran attack in {finish_time - load_time}s')
        print(f'TOTAL TIME: {finish_time - start_time}s')

    
    if args.interactive:
        print('Running in interactive mode')
        print('----------------------------')

        while True:
            print('Enter a sentence to attack or "q" to quit:')
            text = input()

            if text == 'q':
                break

            tokenized_text = TokenizedText(model, text)

            pred = attack._call_model([tokenized_text])
            label = int(pred.argmax())

            print('Attacking...')

            attack.attack([(label, text)])

