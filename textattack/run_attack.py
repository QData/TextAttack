'''
A command line parser to run an attack

'''

import argparse
import time
import os

import textattack.attacks as attacks
import textattack.datasets as datasets
import textattack.models as models
import textattack.constraints as constraints
import textattack.transformations as transformations

from textattack.tokenized_text import TokenizedText

DATASET_CLASS_NAMES = {
    'agnews':           datasets.classification.AGNews,
    'imdb':             datasets.classification.IMDBSentiment,
    'kaggle-fake-news': datasets.classification.KaggleFakeNews,
    'mr':               datasets.classification.MovieReviewSentiment,
    'yelp-sentiment':   datasets.classification.YelpSentiment
}

MODEL_CLASS_NAMES = {
    'bert-yelp-sentiment':  models.classification.bert.BERTForYelpSentimentClassification,
    'cnn-yelp-sentiment':   models.classification.cnn.WordCNNForYelpSentimentClassification,
    'cnn-imdb':             models.classification.cnn.WordCNNForIMDBSentimentClassification,
    'lstm-yelp-sentiment':  models.classification.lstm.LSTMForYelpSentimentClassification,
    'lstm-imdb':            models.classification.lstm.LSTMForIMDBSentimentClassification,
}

MODELS_BY_DATASET = {
    'imdb': ['cnn-imdb', 'lstm-imdb'],
    'yelp-sentiment': ['bert-yelp-sentiment', 'cnn-yelp-sentiment', 'lstm-yelp-sentiment']
}

def get_args():
    parser = argparse.ArgumentParser(description='A commandline parser for TextAttack', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    data_group = parser.add_mutually_exclusive_group(required=True)
    
    parser.add_argument('--attack', type=str, required=False, default='greedy-wir-counterfit',
        choices=['greedy-counterfit', 'ga-counterfit', 'greedy-wir-counterfit'], 
        help='The type of attack to run.')

    parser.add_argument('--transformation', type=str, required=False, default='word-swap-embedding',
        choices=['word-swap-embedding', 'word-swap-homoglyph'],
        help='The type of transformation to apply')
        
    parser.add_argument('--model', type=str, required=False, default='bert-yelp-sentiment',
        choices=MODEL_CLASS_NAMES.keys(), help='The classification model to attack.')
    
    parser.add_argument('--constraints', type=str, required=False, nargs='*',
        default=['use', 'lang-tool'], 
        help=('Constraints to add to the attack. Usage: "--constraints use lang-tool:{threshold} ' 
        'goog-lm" to use the default use similarity of .9 and to choose the threshold'))
    
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
    if args.model not in MODEL_CLASS_NAMES:
        raise ValueError(f'Error: unsupported model {args.model}')
    
    model = MODEL_CLASS_NAMES[args.model]()
    
    # Transformation
    if args.transformation == 'word-swap-embedding':
        transformation = transformations.WordSwapEmbedding()
    elif args.transformation == 'word-swap-homoglyph':
        transformation = transformations.WordSwapHomoglyph()

    # Attacks
    if args.attack == 'greedy-counterfit':
        attack = attacks.GreedyWordSwap(model, transformation)
    elif args.attack == 'ga-counterfit':
        attack = attacks.GeneticAlgorithm(model, transformation)
    elif args.attack == 'greedy-wir-counterfit':
        attack = attacks.GreedyWordSwapWIR(model, transformation)

    # Constraints
    if args.constraints:

        defined_constraints = []

        for constraint in args.constraints:
            if 'use:' in constraint:
                similarity = constraint.replace('use:', '')
                defined_constraints.append(constraints.semantics.UniversalSentenceEncoder(float(similarity), metric='cosine'))
            elif constraint == 'use':
                # Default similarity to .9 if no similarity is given.
                defined_constraints.append(constraints.semantics.UniversalSentenceEncoder(.90, metric='cosine'))

            elif 'lang-tool:' in constraint:
                threshold = constraint.replace('lang-tool:', '')
                defined_constraints.append(constraints.syntax.LanguageTool(float(threshold)))
            elif constraint == 'lang-tool':
                # Default threshold to 0 if no threshold is given
                defined_constraints.append(constraints.syntax.LanguageTool(0))

            elif constraint == 'goog-lm':
                defined_constraints.append(constraints.semantics.google_language_model.GoogleLanguageModel())

            else:
                raise ValueError((f'{constraint} is not a valid constraint. ' 
                    'Valid options are "use", "lang-tool", or "goog-lm". Use "-h" for help.'))

        attack.add_constraints(defined_constraints)
    
    # Data
    if args.data in DATASET_CLASS_NAMES:
        dataset_class = DATASET_CLASS_NAMES[args.data]
        data = dataset_class(args.num_examples)
    else:
        raise ValueError(f'Unknown dataset {args.data}')

    # Output file
    if args.out_file is not None:
        attack.add_output_file(args.out_file)


    load_time = time.time()

    if args.data is not None and not args.interactive:
        check_model_and_data_compatibility(args.data, args.model)
        attack.enable_visdom()
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

            tokenized_text = TokenizedText(model, text)

            pred = attack._call_model([tokenized_text])
            label = int(pred.argmax())

            print('Attacking...')

        attack.enable_visdom()
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

            tokenized_text = TokenizedText(model, text)

            pred = attack._call_model([tokenized_text])
            label = int(pred.argmax())

            print('Attacking...')

            attack.enable_visdom()
            attack.attack([(label, text)])
