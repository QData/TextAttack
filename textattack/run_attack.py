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
    'alzantot':      'textattack.attack_recipes.Alzantot2018GeneticAlgorithm',
    'alz-adjusted':  'textattack.attack_recipes.Alzantot2018GeneticAlgorithmAdjusted',
    'textfooler':    'textattack.attack_recipes.Jin2019TextFooler',
    'tf-adjusted':   'textattack.attack_recipes.Jin2019TextFoolerAdjusted',
}

MODEL_CLASS_NAMES = {
    #
    # Text classification models
    #
    
    # BERT models - default uncased
    'bert-ag-news':             'textattack.models.classification.bert.BERTForAGNewsClassification',
    'bert-imdb':                'textattack.models.classification.bert.BERTForIMDBSentimentClassification',
    'bert-mr':                  'textattack.models.classification.bert.BERTForMRSentimentClassification',
    'bert-yelp-sentiment':      'textattack.models.classification.bert.BERTForYelpSentimentClassification',
    # CNN models
    'cnn-ag-news':              'textattack.models.classification.cnn.WordCNNForAGNewsClassification',
    'cnn-imdb':                 'textattack.models.classification.cnn.WordCNNForIMDBSentimentClassification',
    'cnn-mr':                   'textattack.models.classification.cnn.WordCNNForMRSentimentClassification',
    'cnn-yelp-sentiment':       'textattack.models.classification.cnn.WordCNNForYelpSentimentClassification',
    # LSTM models
    'lstm-ag-news':             'textattack.models.classification.lstm.LSTMForAGNewsClassification',
    'lstm-imdb':                'textattack.models.classification.lstm.LSTMForIMDBSentimentClassification',
    'lstm-mr':                  'textattack.models.classification.lstm.LSTMForMRSentimentClassification',
    'lstm-yelp-sentiment':      'textattack.models.classification.lstm.LSTMForYelpSentimentClassification',
    
    #
    # Textual entailment models
    #
    
    # BERT models
    'bert-mnli':                'textattack.models.entailment.bert.BERTForMNLI',
    'bert-snli':                'textattack.models.entailment.bert.BERTForSNLI',
}

DATASET_BY_MODEL = {
    #
    # Text classification datasets
    #
    
    # AG News
    'bert-ag-news':             datasets.classification.AGNews,
    'cnn-ag-news':              datasets.classification.AGNews,
    'lstm-ag-news':             datasets.classification.AGNews,
    # IMDB 
    'bert-imdb':                datasets.classification.IMDBSentiment,
    'cnn-imdb':                 datasets.classification.IMDBSentiment,
    'lstm-imdb':                datasets.classification.IMDBSentiment,
    # MR
    'bert-mr':                  datasets.classification.MovieReviewSentiment,
    'cnn-mr':                   datasets.classification.MovieReviewSentiment,
    'lstm-mr':                  datasets.classification.MovieReviewSentiment,
    # Yelp
    'bert-yelp-sentiment':      datasets.classification.YelpSentiment,
    'cnn-yelp-sentiment':       datasets.classification.YelpSentiment,
    'lstm-yelp-sentiment':      datasets.classification.YelpSentiment,
    
    #
    # Textual entailment datasets
    #
    'bert-mnli':                datasets.entailment.MNLI,
    'bert-snli':                datasets.entailment.SNLI,
}

TRANSFORMATION_CLASS_NAMES = {
    'word-swap-embedding':             'textattack.transformations.WordSwapEmbedding',
    'word-swap-homoglyph':             'textattack.transformations.WordSwapHomoglyph',
    'word-swap-neighboring-char-swap': 'textattack.transformations.WordSwapNeighboringCharacterSwap',
}

CONSTRAINT_CLASS_NAMES = {
    'embedding':    'textattack.constraints.semantics.WordEmbeddingDistance',
    'goog-lm':      'textattack.constraints.semantics.language_models.GoogleLanguageModel',
    'bert':         'textattack.constraints.semantics.sentence_encoders.BERT',
    'infer-sent':   'textattack.constraints.semantics.sentence_encoders.InferSent',
    'use':          'textattack.constraints.semantics.sentence_encoders.UniversalSentenceEncoder',
    'lang-tool':    'textattack.constraints.syntax.LanguageTool', 
}

ATTACK_CLASS_NAMES = {
    'greedy-word':        'textattack.attacks.blackbox.GreedyWordSwap',
    'ga-word':            'textattack.attacks.blackbox.GeneticAlgorithm',
    'greedy-word-wir':    'textattack.attacks.blackbox.GreedyWordSwapWIR',
}


def get_args():
    parser = argparse.ArgumentParser(description='A commandline parser for TextAttack', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--transformations', type=str, required=False, nargs='*',
        default=['word-swap-embedding'], choices=TRANSFORMATION_CLASS_NAMES.keys(),
        help='The transformations to apply.')
        
    parser.add_argument('--model', type=str, required=False, default='bert-yelp-sentiment',
        choices=MODEL_CLASS_NAMES.keys(), help='The classification model to attack.')
    
    parser.add_argument('--constraints', type=str, required=False, nargs='*',
        default=[], choices=CONSTRAINT_CLASS_NAMES.keys(),
        help=('Constraints to add to the attack. Usage: "--constraints {constraint}:{arg_1}={value_1},{arg_3}={value_3}"'))
    
    parser.add_argument('--out_dir', type=str, required=False, default=None,
        help='A directory to output results to.')
    
    parser.add_argument('--enable_visdom', action='store_true',
        help='Enable logging to visdom.')
    
    parser.add_argument('--disable_stdout', action='store_true',
        help='Disable logging to stdout')
   
    parser.add_argument('--enable_csv', nargs='?', default=None, const='fancy', type=str,
        help='Enable logging to csv. Use --enable_csv plain to remove [[]] around words.')

    parser.add_argument('--num_examples', '-n', type=int, required=False, 
        default='5', help='The number of examples to process.')
    
    parser.add_argument('--num_examples_offset', '-o', type=int, required=False, 
        default=0, help='The offset to start at in the dataset.')
  
    parser.add_argument('--attack_n', action='store_true',
        help='Attack n examples, not counting examples where the model is initially wrong.')

    parser.add_argument('--shuffle', action='store_true', required=False, 
        default=False, help='Randomly shuffle the data before attacking')
    
    parser.add_argument('--interactive', action='store_true', default=False,
        help='Whether to run attacks interactively.')
    
    attack_group = parser.add_mutually_exclusive_group(required=False)
    
    attack_group.add_argument('--attack', type=str, required=False, default='greedy-word-wir', 
        help='The type of attack to run.', choices=ATTACK_CLASS_NAMES.keys())
    
    attack_group.add_argument('--recipe', type=str, required=False, default=None, 
        help='full attack recipe (overrides provided transformation & constraints)')
    
    args = parser.parse_args()
    
    return args

def parse_transformation_from_args():
    # Transformations
    _transformations = []    
    for transformation in args.transformations:
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
    if ':' in args.recipe:
        recipe_name, params = args.recipe.split(':')
        if recipe_name not in RECIPE_NAMES:
            raise ValueError(f'Error: unsupported recipe {recipe_name}')
        recipe = eval(f'{RECIPE_NAMES[recipe_name]}(model, {params})')
    elif args.recipe in RECIPE_NAMES:
        recipe = eval(f'{RECIPE_NAMES[args.recipe]}(model)')
    else:
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
    # Cache TensorFlow Hub models here, if not otherwise specified.
    if 'TFHUB_CACHE_DIR' not in os.environ:
        os.environ['TFHUB_CACHE_DIR'] = './tensorflow-hub'
    
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

    out_time = int(time.time()*1000) # Output file
    if args.out_dir is not None:
        outfile_name = 'attack-{}.txt'.format(out_time)
        attack.add_output_file(os.path.join(args.out_dir, outfile_name))

    # csv
    if args.enable_csv:
        out_dir = args.out_dir if args.out_dir else 'outputs'
        outfile_name = 'attack-{}.csv'.format(out_time)
        plain = args.enable_csv == 'plain'
        attack.add_output_csv(os.path.join(out_dir, outfile_name), plain)

    # Visdom
    if args.enable_visdom:
        attack.enable_visdom()

    # Stdout
    if not args.disable_stdout:
        attack.enable_stdout()

    load_time = time.time()
    print(f'Loaded in {load_time - start_time}s')

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

            tokenized_text = TokenizedText(text, model.tokenizer)

            pred = attack._call_model([tokenized_text])
            label = int(pred.argmax())

            print('Attacking...')

            attack.attack([(label, text)])
    
    else:
        # Not interactive? Use default dataset.
        if args.model in DATASET_BY_MODEL:
            data = DATASET_BY_MODEL[args.model](offset=args.num_examples_offset)
        else:
            raise ValueError(f'Error: unsupported model {args.model}')
            
        attack.attack(data, num_examples=args.num_examples,
                      attack_n=args.attack_n, shuffle=args.shuffle)

        finish_time = time.time()

        print(f'Total time: {finish_time - start_time}s')
