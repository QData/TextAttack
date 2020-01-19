"""
A command line parser to run an attack from user specifications.
"""

import argparse
import functools
import os
import time
import torch
import tqdm

import textattack.attack_recipes as attack_recipes
import textattack.attack_methods as attack_methods
import textattack.constraints as constraints
import textattack.datasets as datasets
import textattack.models as models
import textattack.transformations as transformations

from textattack.shared.tokenized_text import TokenizedText


RECIPE_NAMES = {
    'alzantot':      'attack_recipes.Alzantot2018GeneticAlgorithm',
    'alz-adjusted':  'attack_recipes.Alzantot2018GeneticAlgorithmAdjusted',
    'textfooler':    'attack_recipes.Jin2019TextFooler',
    'tf-adjusted':   'attack_recipes.Jin2019TextFoolerAdjusted',
}

MODEL_CLASS_NAMES = {
    #
    # Text classification models
    #
    
    # BERT models - default uncased
    'bert-ag-news':             'models.classification.bert.BERTForAGNewsClassification',
    'bert-imdb':                'models.classification.bert.BERTForIMDBSentimentClassification',
    'bert-mr':                  'models.classification.bert.BERTForMRSentimentClassification',
    'bert-yelp-sentiment':      'models.classification.bert.BERTForYelpSentimentClassification',
    # CNN models
    'cnn-ag-news':              'models.classification.cnn.WordCNNForAGNewsClassification',
    'cnn-imdb':                 'models.classification.cnn.WordCNNForIMDBSentimentClassification',
    'cnn-mr':                   'models.classification.cnn.WordCNNForMRSentimentClassification',
    'cnn-yelp-sentiment':       'models.classification.cnn.WordCNNForYelpSentimentClassification',
    # LSTM models
    'lstm-ag-news':             'models.classification.lstm.LSTMForAGNewsClassification',
    'lstm-imdb':                'models.classification.lstm.LSTMForIMDBSentimentClassification',
    'lstm-mr':                  'models.classification.lstm.LSTMForMRSentimentClassification',
    'lstm-yelp-sentiment':      'models.classification.lstm.LSTMForYelpSentimentClassification',
    
    #
    # Textual entailment models
    #
    
    # BERT models
    'bert-mnli':                'models.entailment.bert.BERTForMNLI',
    'bert-snli':                'models.entailment.bert.BERTForSNLI',
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
    'word-swap-embedding':             'transformations.WordSwapEmbedding',
    'word-swap-homoglyph':             'transformations.WordSwapHomoglyph',
    'word-swap-neighboring-char-swap': 'transformations.WordSwapNeighboringCharacterSwap',
}

CONSTRAINT_CLASS_NAMES = {
    'embedding':    'constraints.semantics.WordEmbeddingDistance',
    'goog-lm':      'constraints.semantics.language_models.GoogleLanguageModel',
    'bert':         'constraints.semantics.sentence_encoders.BERT',
    'infer-sent':   'constraints.semantics.sentence_encoders.InferSent',
    'use':          'constraints.semantics.sentence_encoders.UniversalSentenceEncoder',
    'lang-tool':    'constraints.syntax.LanguageTool', 
}

ATTACK_CLASS_NAMES = {
    'greedy-word':        'attack_methods.GreedyWordSwap',
    'ga-word':            'attack_methods.GeneticAlgorithm',
    'greedy-word-wir':    'attack_methods.GreedyWordSwapWIR',
}


def get_args():
    parser = argparse.ArgumentParser(description='A commandline parser for TextAttack', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--transformation', type=str, required=False,
        default='word-swap-embedding', choices=TRANSFORMATION_CLASS_NAMES.keys(),
        help='The transformation to apply.')
        
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
        help='Enable logging to csv. Options are "fancy" and "plain". Use --enable_csv plain to remove [[]] around words.')

    parser.add_argument('--num_examples', '-n', type=int, required=False, 
        default='5', help='The number of examples to process.')
    
    parser.add_argument('--num_examples_offset', '-o', type=int, required=False, 
        default=0, help='The offset to start at in the dataset.')
  
    parser.add_argument('--attack_n', action='store_true',
        help='Attack n examples, not counting examples where the model is initially wrong.')

    parser.add_argument('--shuffle', action='store_true', required=False, 
        default=False, help='Randomly shuffle the data before attacking')
    
    threading_group = parser.add_mutually_exclusive_group(required=False)
    threading_group.add_argument('--interactive', action='store_true', default=False,
        help='Whether to run attacks interactively.')
    threading_group.add_argument('--num_threads', default=1, type=int, help='Run attack across multiple threads.')
    
    attack_group = parser.add_mutually_exclusive_group(required=False)
    
    attack_group.add_argument('--attack', type=str, required=False, default='greedy-word-wir', 
        help='The type of attack to run.', choices=ATTACK_CLASS_NAMES.keys())
    
    attack_group.add_argument('--recipe', type=str, required=False, default=None, 
        help='full attack recipe (overrides provided transformation & constraints)')
    
    args = parser.parse_args()
    
    return args

def parse_transformation_from_args(args):
    # Transformations
    _transformations = []    
    transformation = args.transformation
    if ':' in transformation:
        transformation_name, params = transformation.split(':')
        if transformation_name not in TRANSFORMATION_CLASS_NAMES:
            raise ValueError(f'Error: unsupported transformation {transformation_name}')
        _transformations.append(eval(f'{TRANSFORMATION_CLASS_NAMES[transformation_name]}({params})'))
    elif transformation in TRANSFORMATION_CLASS_NAMES:
        _transformations.append(eval(f'{TRANSFORMATION_CLASS_NAMES[transformation]}()'))
    else:
        raise ValueError(f'Error: unsupported transformation {transformation}')
    return _transformations[0]

def parse_constraints_from_args(args):
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

def parse_recipe_from_args(args):
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

def parse_attack_from_args(model, _transformation, args):
    if ':' in args.attack:
        attack_name, params = args.attack.split(':')
        if attack_name not in ATTACK_CLASS_NAMES:
            raise ValueError(f'Error: unsupported attack {attack_name}')
        attack = eval(f'{ATTACK_CLASS_NAMES[attack_name]}(model, _transformation, {params})')
    elif args.attack in ATTACK_CLASS_NAMES:
        attack = eval(f'{ATTACK_CLASS_NAMES[args.attack]}(model, _transformation)')
    else:
        raise ValueError(f'Error: unsupported attack {args.attack}')
    return attack

def parse_model_from_args(args):
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

@functools.lru_cache(maxsize=None)
def get_model_and_attack(worker_id):
    model = parse_model_from_args(args)
    
    # Distribute workload across GPUs.
    num_gpus_available = max(torch.cuda.device_count(), 1)
    gpu_id = num_gpus_available % worker_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
    if args.recipe:
        attack = parse_recipe_from_args(args)
    else:
        _transformation = parse_transformation_from_args(args)
        attack = parse_attack_from_args(model, _transformation, args)
        attack.add_constraints(parse_constraints_from_args(args))
    
    # System time provides a identifier for the output file.
    out_time = int(time.time() * 1000) 
    if args.out_dir is not None:
        outfile_name = 'attack-{}.txt'.format(out_time)
        attack.add_output_file(os.path.join(args.out_dir, outfile_name))
    
    # Log attack results to CSV.
    if args.enable_csv:
        out_dir = args.out_dir if args.out_dir else 'outputs'
        outfile_name = 'attack-{}.csv'.format(out_time)
        plain = args.enable_csv == 'plain'
        attack.add_output_csv(os.path.join(out_dir, outfile_name), plain)
    
    # Print attack results to visdom.
    if args.enable_visdom:
        attack.enable_visdom()
    
    # Print attack results to standard out.
    if not args.disable_stdout:
        attack.enable_stdout()
    
    return model, attack

def initialize_attack(worker_id):
    worker_id = torch.multiprocessing.current_process()._identity[0]
    model, attack = get_model_and_attack(worker_id)
    return

def attack_sample(original_label, text):
    worker_id = torch.multiprocessing.current_process()._identity[0]
    model, attack = get_model_and_attack(worker_id)
    tokenized_text = TokenizedText(text, attack.tokenizer)
    predicted_label = attack._call_model([tokenized_text])[0].argmax().item()
    # @TODO return SkippedAttackResult, if predicted != original
    return attack.attack_one(predicted_label, tokenized_text)

def main(args):
    # Disable tensorflow logs, except in the case of an error.
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Cache TensorFlow Hub models here, if not otherwise specified.
    if 'TFHUB_CACHE_DIR' not in os.environ:
        os.environ['TFHUB_CACHE_DIR'] = '~/.cache/tensorflow-hub'
    
    start_time = time.time()
    
    load_time = time.time()

    # Use default dataset for chosen model.
    if args.model in DATASET_BY_MODEL:
        dataset = DATASET_BY_MODEL[args.model](offset=args.num_examples_offset)
    else:
        raise ValueError(f'Error: no default dataset for model {args.model}')
    
    # Initialize workers across `args.num_threads`. Interactive mode uses just 1 thread.
    if args.num_threads > 1:
        print(f'Running attack with {args.num_threads} threads.')
    
    # Initialize attacker pool - one attack per thread.
    attacker_pool = torch.multiprocessing.Pool(processes=args.num_threads)
    dummy_args = list((x,) for x in range(args.num_threads))
    attacker_pool.starmap(initialize_attack, dummy_args)
    # attacker_pool.close()
    # attacker_pool.join()
    
    # Get first attack to use as a logger.
    model, attack = get_model_and_attack(2)
    
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
        attack.log_attack_start()
        
        # Feed the dataset to the attacker pool.
        pbar = tqdm.tqdm(total=args.num_examples)
        def _update_tqdm(*_):
            pbar.update()
        
        results = []
        for args in range(args.num_examples): # @TODO support `attack_n`
            args = next(dataset)
            result = attacker_pool.apply_async(attack_sample, args=args, callback=_update_tqdm)
            results.append(result)
        attacker_pool.close()
        attacker_pool.join()
        pbar.close()
        
        # Log results using the first attack logger.
        for result in results:
            attack.logger.log_result(result.get())
        attack.log_attack_end()
        
    finish_time = time.time()

    print(f'Total time: {finish_time - start_time}s')


if __name__ == '__main__':
    args = torch.multiprocessing.Manager().Namespace(
        **vars(get_args())
    )
    main(args)