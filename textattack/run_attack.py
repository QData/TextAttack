'''
A command line parser to run an attack
'''

import argparse
import time
import os

import textattack.datasets as datasets
import textattack.attacks as attacks
import textattack.models as models
import textattack.constraints as constraints
import textattack.transformations as transformations

from textattack.tokenized_text import TokenizedText


MODEL_CLASS_NAMES = {
    #
    # BERT models - default uncased
    #
    'bert-imdb':                'models.classification.bert.BERTForIMDBSentimentClassification',
    'bert-mr':                  'models.classification.bert.BERTForMRSentimentClassification',
    'bert-yelp-sentiment':      'models.classification.bert.BERTForYelpSentimentClassification',
    #
    # CNN models
    #
    'cnn-imdb':                 'models.classification.cnn.WordCNNForIMDBSentimentClassification',
    'cnn-mr':                   'models.classification.cnn.WordCNNForMRSentimentClassification',
    'cnn-yelp-sentiment':       'models.classification.cnn.WordCNNForYelpSentimentClassification',
    #
    # LSTM models
    #
    'lstm-imdb':                'models.classification.lstm.LSTMForIMDBSentimentClassification',
    'lstm-mr':                  'models.classification.lstm.LSTMForMRSentimentClassification',
    'lstm-yelp-sentiment':      'models.classification.lstm.LSTMForYelpSentimentClassification',
}

DATASET_BY_MODEL = {
    #
    # IMDB models 
    #
    'bert-imdb':                datasets.classification.IMDBSentiment,
    'cnn-imdb':                 datasets.classification.IMDBSentiment,
    'lstm-imdb':                datasets.classification.IMDBSentiment,
    #
    # MR models
    #
    'bert-mr':                  datasets.classification.MovieReviewSentiment,
    'cnn-mr':                   datasets.classification.MovieReviewSentiment,
    'lstm-mr':                  datasets.classification.MovieReviewSentiment,
    #
    # Yelp models
    #
    'bert-yelp-sentiment':      datasets.classification.YelpSentiment,
    'cnn-yelp-sentiment':       datasets.classification.YelpSentiment,
    'lstm-yelp-sentiment':      datasets.classification.YelpSentiment,
}

TRANSFORMATION_CLASS_NAMES = {
    'word-swap-embedding':  'transformations.WordSwapEmbedding',
    'word-swap-homoglyph':  'transformations.WordSwapHomoglyph',
    'word-swap-neighboring-char-swap': 'transformations.WordSwapNeighboringCharacterSwap',
}

CONSTRAINT_CLASS_NAMES = {
    'use':          'constraints.semantics.UniversalSentenceEncoder',
    'lang-tool':    'constraints.syntax.LanguageTool', 
    'goog-lm':      'constraints.semantics.google_language_model.GoogleLanguageModel',
}

ATTACK_CLASS_NAMES = {
    'greedy-word':        'attacks.blackbox.GreedyWordSwap',
    'ga-word':            'attacks.blackbox.GeneticAlgorithm',
    'greedy-word-wir':    'attacks.blackbox.GreedyWordSwapWIR',
}


def get_args():
    parser = argparse.ArgumentParser(description='A commandline parser for TextAttack', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--attack', type=str, required=False, default='greedy-word-wir', 
        help='The type of attack to run.')

    parser.add_argument('--transformation', type=str, required=False, nargs='*',
        default=['word-swap-embedding'],
        help='The type of transformation to apply')
        
    parser.add_argument('--model', type=str, required=False, default='bert-yelp-sentiment',
        choices=MODEL_CLASS_NAMES.keys(), help='The classification model to attack.')
    
    parser.add_argument('--constraints', type=str, required=False, nargs='*',
        default=['use'], 
        help=('Constraints to add to the attack. Usage: "--constraints {constraint}:{arg_1}={value_1},{arg_3}={value_3}' 
        ' Options are use, lang-tool, and goog-lm'))
    
    parser.add_argument('--out_file', type=str, required=False,
        help='The file to output the results to.')
    
    parser.add_argument('--num_examples', type=int, required=False, 
        default='5', help='The number of examples to attack.')
    
    parser.add_argument('--interactive', action='store_true', 
        help='Whether to run attacks interactively.')
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_args()
    
    # Only use one GPU, if we have one.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Disable tensorflow logs, except in the case of an error.
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    start_time = time.time()

    # Models
    if ':' in args.model:
        model_name, params = args.model.split(':')
        if model_name not in MODEL_CLASS_NAMES:
            raise ValueError(f'Error: unsupported model {model_name}')
        model = eval(f'{MODEL_CLASS_NAMES[model_name]}({params})')
    elif args.model in MODEL_CLASS_NAMES:
        model = eval(f'{MODEL_CLASS_NAMES[args.model]}()')
    else: 
        raise ValueError(f'Error: unsupported model {args.model}')
    
    # Transformations
    defined_transformations = []    
    for transformation in args.transformation:
        if ':' in transformation:
            transformation_name, params = transformation.split(':')
            if transformation_name not in TRANSFORMATION_CLASS_NAMES:
                raise ValueError(f'Error: unsupported transformation {transformation_name}')
            defined_transformations.append(eval(f'{TRANSFORMATION_CLASS_NAMES[transformation_name]}({params})'))
        elif transformation in TRANSFORMATION_CLASS_NAMES:
            defined_transformations.append(eval(f'{TRANSFORMATION_CLASS_NAMES[transformation]}()'))
        else:
            raise ValueError(f'Error: unsupported transformation {transformation}')

    # Attacks
    if ':' in args.attack:
        attack_name, params = args.attack.split(':')
        if attack_name not in ATTACK_CLASS_NAMES:
            raise ValueError(f'Error: unsupported attack {attack_name}')
        attack = eval(f'{ATTACK_CLASS_NAMES[attack_name]}(model, defined_transformations, {params})')
    elif args.attack in ATTACK_CLASS_NAMES:
        attack = eval(f'{ATTACK_CLASS_NAMES[args.attack]}(model, defined_transformations)')
    else:
        raise ValueError(f'Error: unsupported attack {args.attack}')

    # Constraints
    if args.constraints:
        defined_constraints = []
        for constraint in args.constraints:
            if ':' in constraint:
                constraint_name, params = constraint.split(':')
                if constraint_name not in CONSTRAINT_CLASS_NAMES:
                    raise ValueError(f'Error: unsupported constraint {constraint_name}')
                defined_constraints.append(eval(f'{CONSTRAINT_CLASS_NAMES[constraint_name]}({params})'))
            elif constraint in CONSTRAINT_CLASS_NAMES:
                defined_constraints.append(eval(f'{CONSTRAINT_CLASS_NAMES[constraint]}()'))
            else:
                raise ValueError(f'Error: unsupported constraint {constraint}')

        attack.add_constraints(defined_constraints)
    
    # Data
    if not args.interactive:
        if args.model in DATASET_BY_MODEL:
            data = DATASET_BY_MODEL[args.model](n=args.num_examples)
        else:
            raise ValueError(f'Error: unsupported model {args.model}')


    # Output file
    if args.out_file is not None:
        attack.add_output_file(args.out_file)


    load_time = time.time()

    if not args.interactive:
        
        data_name = args.model.split('-', 1)[1]
        print(f'Model: {args.model} / Dataset: {data_name}')
        
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
            
            if not text:
                continue

            tokenized_text = TokenizedText(text, model.convert_text_to_ids)

            pred = attack._call_model([tokenized_text])
            label = int(pred.argmax())

            print('Attacking...')

            attack.attack([(label, text)])
