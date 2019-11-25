"""
A command line parser to run an attack from user specifications.
"""

import argparse
import time
import os

import textattack.attack_recipes as attack_recipes
import textattack.attacks as attacks
import textattack.constraints as constraints
import textattack.datasets as datasets
import textattack.models as models
import textattack.transformations as transformations

from textattack.tokenized_text import TokenizedText


RECIPE_NAMES = {
    'alzantot':     attack_recipes.Alzantot2018GeneticAlgorithm,
    'textfooler':   attack_recipes.Jin2019TextFooler
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
    'word-swap-embedding':             'transformations.WordSwapEmbedding',
    'word-swap-homoglyph':             'transformations.WordSwapHomoglyph',
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

    parser.add_argument('--transformation', type=str, required=False, nargs='*',
        default=['word-swap-embedding'],
        help='The type of transformation to apply')
        
    parser.add_argument('--model', type=str, required=False, default='bert-yelp-sentiment',
        choices=MODEL_CLASS_NAMES.keys(), help='The classification model to attack.')
    
    parser.add_argument('--constraints', type=str, required=False, nargs='*',
        default=['use'], 
        help=('Constraints to add to the attack. Usage: "--constraints {constraint}:{arg_1}={value_1},{arg_3}={value_3}' 
        ' Options are use, lang-tool, and goog-lm'))
    
    parser.add_argument('--out_dir', type=str, required=False, default=None,
        help='A directory to output results to.')
    
    parser.add_argument('--enable_visdom', action='store_true', default=False,
        help='Enable logging to visdom.')
    
    parser.add_argument('--num_examples', '--n', type=int, required=False, 
        default='5', help='The number of examples to attack.')
    
    parser.add_argument('--num_examples_offset', '--o', type=int, required=False, 
        default=0, help='The offset to start at in the dataset.')
    
    parser.add_argument('--shuffle', action='store_true', required=False, 
        default=False, help='Randomly shuffle the data before attacking')
    
    parser.add_argument('--interactive', action='store_true', default=False,
        help='Whether to run attacks interactively.')
    
    attack_group = parser.add_mutually_exclusive_group(required=False)
    
    attack_group.add_argument('--attack', type=str, required=False, default='greedy-word-wir', 
        help='The type of attack to run.')
    
    attack_group.add_argument('--recipe', type=str, required=False, default=None, 
        help='full attack recipe (overrides provided transformation & constraints)')
    
    args = parser.parse_args()
    
    return args

def parse_transformation_from_args():
    # Transformations
    _transformations = []    
    for transformation in args.transformation:
        if ':' in transformation:
            transformation_name, params = transformation.split(':')
            if transformation_name not in TRANSFORMATION_CLASS_NAMES:
                raise ValueError(f'Error: unsupported transformation {transformation_name}')
            _transformations.append(eval(f'{TRANSFORMATION_CLASS_NAMES[transformation_name]}({params})'))
        elif transformation in TRANSFORMATION_CLASS_NAMES:
            _transformations.append(eval(f'{TRANSFORMATION_CLASS_NAMES[transformation]}()'))
        else:
            raise ValueError(f'Error: unsupported transformation {transformation}')
    return _transformations

def parse_constraints_from_args():
    # Constraints
    if not args.constraints: 
        return []
    
    _constraints = []
    for constraint in args.constraints:
        if ':' in constraint:
            constraint_name, params = constraint.split(':')
            if constraint_name not in CONSTRAINT_CLASS_NAMES:
                raise ValueError(f'Error: unsupported constraint {constraint_name}')
            _constraints.append(eval(f'{CONSTRAINT_CLASS_NAMES[constraint_name]}({params})'))
        elif constraint in CONSTRAINT_CLASS_NAMES:
            _constraints.append(eval(f'{CONSTRAINT_CLASS_NAMES[constraint]}()'))
        else:
            raise ValueError(f'Error: unsupported constraint {constraint}')
    
    return _constraints

def parse_recipe_from_args():
    try:
        recipe = RECIPE_NAMES[args.recipe](model)
    except KeyError:
        raise Error('Invalid recipe {args.recipe}')
    return recipe

def parse_attack_from_args():
    if ':' in args.attack:
        attack_name, params = args.attack.split(':')
        if attack_name not in ATTACK_CLASS_NAMES:
            raise ValueError(f'Error: unsupported attack {attack_name}')
        attack = eval(f'{ATTACK_CLASS_NAMES[attack_name]}(model, _transformations, {params})')
    elif args.attack in ATTACK_CLASS_NAMES:
        attack = eval(f'{ATTACK_CLASS_NAMES[args.attack]}(model, _transformations)')
    else:
        raise ValueError(f'Error: unsupported attack {args.attack}')
    return attack

def parse_model_from_args():
    if ':' in args.model:
        model_name, params = args.model.split(':')
        if model_name not in MODEL_CLASS_NAMES:
            raise ValueError(f'Error: unsupported model {model_name}')
        model = eval(f'{MODEL_CLASS_NAMES[model_name]}({params})')
    elif args.model in MODEL_CLASS_NAMES:
        model = eval(f'{MODEL_CLASS_NAMES[args.model]}()')
    else: 
        raise ValueError(f'Error: unsupported model {args.model}')
    return model

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
    model = parse_model_from_args()
    
    if args.recipe:
        attack = parse_recipe_from_args()
    else:
        # Transformations
        _transformations = parse_transformation_from_args()
        # Attack
        attack = parse_attack_from_args()
        attack.add_constraints(parse_constraints_from_args())

    # Output file
    if args.out_dir is not None:
        outfile_name = 'attack-{}.txt'.format(int(time.time()))
        attack.add_output_file(os.path.join(args.out_dir, outfile_name))

    # Visdom
    if args.enable_visdom:
        attack.enable_visdom()

    load_time = time.time()

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

            tokenized_text = TokenizedText(model, text)

            pred = attack._call_model([tokenized_text])
            label = int(pred.argmax())

            print('Attacking...')

            attack.attack([(label, text)])
    
    else:
        # Not interactive? Use default dataset.
        if args.model in DATASET_BY_MODEL:
            data = DATASET_BY_MODEL[args.model](n=args.num_examples)
        else:
            raise ValueError(f'Error: unsupported model {args.model}')
            
        data_name = args.model.split('-', 1)[1]
        print(f'Model: {args.model} / Dataset: {data_name}')
        
        attack.attack(data, shuffle=args.shuffle)

        finish_time = time.time()

        print(f'Loaded in {load_time - start_time}s')
        print(f'Ran attack in {finish_time - load_time}s')
        print(f'TOTAL TIME: {finish_time - start_time}s')
