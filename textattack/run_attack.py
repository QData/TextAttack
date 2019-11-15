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

DATASET_CLASS_NAMES = {
    'agnews':           datasets.classification.AGNews,
    'imdb':             datasets.classification.IMDBSentiment,
    'kaggle-fake-news': datasets.classification.KaggleFakeNews,
    'mr':               datasets.classification.MovieReviewSentiment,
    'yelp-sentiment':   datasets.classification.YelpSentiment,
}

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

MODELS_BY_DATASET = {
    'imdb':             ['bert-imdb', 'cnn-imdb', 'lstm-imdb'],
    'mr':               ['bert-mr', 'cnn-mr', 'lstm-mr'],
    'yelp-sentiment':   ['bert-yelp-sentiment', 'cnn-yelp-sentiment', 'lstm-yelp-sentiment']
}

TRANSFORMATION_CLASS_NAMES = {
    'word-swap-embedding':  'transformations.WordSwapEmbedding',
    'word-swap-homoglyph':  'transformations.WordSwapHomoglyph',
}

CONSTRAINT_CLASS_NAMES = {
    'use':          'constraints.semantics.UniversalSentenceEncoder',
    'lang-tool':    'constraints.syntax.LanguageTool', 
    'goog-lm':      'constraints.semantics.google_language_model.GoogleLanguageModel',
}

ATTACK_CLASS_NAMES = {
    'greedy-counterfit':        attacks.blackbox.GreedyWordSwap,
    'ga-counterfit':            attacks.blackbox.GeneticAlgorithm,
    'greedy-wir-counterfit':    attacks.blackbox.GreedyWordSwapWIR,
}


def get_args():
    parser = argparse.ArgumentParser(description='A commandline parser for TextAttack', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    data_group = parser.add_mutually_exclusive_group(required=True)
    
    parser.add_argument('--attack', type=str, required=False, default='greedy-wir-counterfit', 
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
    
    data_group.add_argument('--interactive', action='store_true', 
        help='Whether to run attacks interactively.')
    
    data_group.add_argument('--data', type=str, default='yelp-sentiment',
        choices=DATASET_CLASS_NAMES.keys(), help='The dataset to use.')
    
    args = parser.parse_args()
    
    return args

def check_model_and_data_compatibility(data_name, model_name):
    """
        Prints a warning message if the user attacks a model using data different
        than what it was trained on.
    """
    if not model_name or not data_name:
        return
    elif data_name not in MODELS_BY_DATASET:
        print('Warning: No known models for this dataset.')
    elif model_name not in MODELS_BY_DATASET[data_name]:
        print(f'Warning: model {model_name} incompatible with dataset {data_name}.')


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
    # if ':' in args.attack:
    #     attack_name, params = args.attack.split(':')
    #     if attack_name not in ATTACK_CLASS_NAMES:
    #         raise ValueError(f'Error: unsupported attack {attack_name}')
    #     attack = eval(f'{ATTACK_CLASS_NAMES[attack_name]}({model}, {defined_transformations}, {params})')
    # elif args.attack in ATTACK_CLASS_NAMES:
    #     attack = eval(f'{ATTACK_CLASS_NAMES[args.attack]}({model}, {defined_transformations})')
    # else:
    #      raise ValueError(f'Error: unsupported attack {args.attack}')

    if args.attack in ATTACK_CLASS_NAMES:
        attack = ATTACK_CLASS_NAMES[args.attack](model, defined_transformations)
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
    if args.data in DATASET_CLASS_NAMES:
        dataset_class = DATASET_CLASS_NAMES[args.data]
        data = dataset_class(args.num_examples)

    # Output file
    if args.out_file is not None:
        attack.add_output_file(args.out_file)


    load_time = time.time()

    if args.data is not None and not args.interactive:
        check_model_and_data_compatibility(args.data, args.model)
        
        print(f'Model: {args.model} / Dataset: {args.data}')
        
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
