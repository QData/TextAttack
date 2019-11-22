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

DATASET_CLASS_NAMES = {
    'agnews':           datasets.classification.AGNews,
    'imdb':             datasets.classification.IMDBSentiment,
    'kaggle-fake-news': datasets.classification.KaggleFakeNews,
    'mr':               datasets.classification.MovieReviewSentiment,
    'yelp-sentiment':   datasets.classification.YelpSentiment,
}

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

MODELS_BY_DATASET = {
    'imdb':             ['bert-imdb', 'cnn-imdb', 'lstm-imdb'],
    'mr':               ['bert-mr', 'cnn-mr', 'lstm-mr'],
    'yelp-sentiment':   ['bert-yelp-sentiment', 'cnn-yelp-sentiment', 'lstm-yelp-sentiment']
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
    
    parser.add_argument('--num_examples', '--n', type=int, required=False, 
        default='5', help='The number of examples to attack.')
    
    parser.add_argument('--num_examples_offset', '--o', type=int, required=False, 
        default=0, help='The offset to start at in the dataset.')
    
    parser.add_argument('--shuffle', action='store_true', required=False, 
        default=False, help='Randomly shuffle the data before attacking')
    
    data_group = parser.add_mutually_exclusive_group(required=False)
    
    data_group.add_argument('--interactive', action='store_true', default=False,
        help='Whether to run attacks interactively.')
    
    data_group.add_argument('--data', type=str, default=None,
        choices=DATASET_CLASS_NAMES.keys(), help='The dataset to use.')
    
    attack_group = parser.add_mutually_exclusive_group(required=False)
    
    attack_group.add_argument('--attack', type=str, required=False, default='greedy-word-wir', 
        help='The type of attack to run.')
    
    attack_group.add_argument('--recipe', type=str, required=False, default=None, 
        help='full attack recipe (overrides provided transformation & constraints)')
    
    args = parser.parse_args()
    
    # Default to interactive mode if no dataset specified.
    if not args.data: args.interactive = True
    
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

    load_time = time.time()

    if args.data is not None and not args.interactive:
        check_model_and_data_compatibility(args.data, args.model)
        
        # Data
        dataset_class = DATASET_CLASS_NAMES[args.data]
        data = dataset_class(n=args.num_examples, offset=args.num_examples_offset)
        
        print(f'Model: {args.model} / Dataset: {args.data}')
        
        attack.attack(data, shuffle=args.shuffle)

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
