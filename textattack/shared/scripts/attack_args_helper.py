import argparse
import importlib
import numpy as np
import os
import random
import sys
import textattack
import time
import torch
import pickle
import copy

RECIPE_NAMES = {
    'alzantot':         'textattack.attack_recipes.Alzantot2018',
    'deepwordbug':      'textattack.attack_recipes.DeepWordBugGao2018',
    'hotflip':          'textattack.attack_recipes.HotFlipEbrahimi2017',
    'kuleshov':         'textattack.attack_recipes.Kuleshov2017',
    'seq2sick':         'textattack.attack_recipes.Seq2SickCheng2018BlackBox',
    'textbugger':       'textattack.attack_recipes.TextBuggerLi2018',
    'textfooler':       'textattack.attack_recipes.TextFoolerJin2019',
}

TEXTATTACK_MODEL_CLASS_NAMES = {
    #
    # Text classification models
    #
    
    # BERT models - default uncased
    'bert-base-uncased-ag-news':    'textattack.models.classification.bert.BERTForAGNewsClassification',
    'bert-base-uncased-imdb':       'textattack.models.classification.bert.BERTForIMDBSentimentClassification',
    'bert-base-uncased-mr':         'textattack.models.classification.bert.BERTForMRSentimentClassification',
    'bert-base-uncased-yelp':       'textattack.models.classification.bert.BERTForYelpSentimentClassification',
    # CNN models
    'cnn-ag-news':                  'textattack.models.classification.cnn.WordCNNForAGNewsClassification',
    'cnn-imdb':                     'textattack.models.classification.cnn.WordCNNForIMDBSentimentClassification',
    'cnn-mr':                       'textattack.models.classification.cnn.WordCNNForMRSentimentClassification',
    'cnn-yelp-sentiment':           'textattack.models.classification.cnn.WordCNNForYelpSentimentClassification',
    # LSTM models
    'lstm-ag-news':                 'textattack.models.classification.lstm.LSTMForAGNewsClassification',
    'lstm-imdb':                    'textattack.models.classification.lstm.LSTMForIMDBSentimentClassification',
    'lstm-mr':                      'textattack.models.classification.lstm.LSTMForMRSentimentClassification',
    'lstm-yelp-sentiment':          'textattack.models.classification.lstm.LSTMForYelpSentimentClassification',
    #
    # Textual entailment models
    #
    # BERT models
    'bert-base-uncased-mnli':       'textattack.models.entailment.bert.BERTForMNLI',
    'bert-base-uncased-snli':       'textattack.models.entailment.bert.BERTForSNLI',
    #
    # Translation models
    #
    't5-en2fr':                     'textattack.models.translation.t5.T5EnglishToFrench',
    't5-en2de':                     'textattack.models.translation.t5.T5EnglishToGerman',
    't5-en2ro':                     'textattack.models.translation.t5.T5EnglishToRomanian',
    #
    # Summarization models
    #
    't5-summ':                      'textattack.models.summarization.T5Summarization',
}

DATASET_BY_MODEL = {
    #
    # Text classification datasets
    #
    # AG News
    'bert-base-uncased-ag-news':    textattack.datasets.classification.AGNews,
    'cnn-ag-news':                  textattack.datasets.classification.AGNews,
    'lstm-ag-news':                 textattack.datasets.classification.AGNews,
    # IMDB 
    'bert-base-uncased-imdb':       textattack.datasets.classification.IMDBSentiment,
    'cnn-imdb':                     textattack.datasets.classification.IMDBSentiment,
    'lstm-imdb':                    textattack.datasets.classification.IMDBSentiment,
    # MR
    'bert-base-uncased-mr':         textattack.datasets.classification.MovieReviewSentiment,
    'cnn-mr':                       textattack.datasets.classification.MovieReviewSentiment,
    'lstm-mr':                      textattack.datasets.classification.MovieReviewSentiment,
    # Yelp
    'bert-base-uncased-yelp':       textattack.datasets.classification.YelpSentiment,
    'cnn-yelp-sentiment':           textattack.datasets.classification.YelpSentiment,
    'lstm-yelp-sentiment':          textattack.datasets.classification.YelpSentiment,
    #
    # Textual entailment datasets
    #
    'bert-base-uncased-mnli':       textattack.datasets.entailment.MNLI,
    'bert-base-uncased-snli':       textattack.datasets.entailment.SNLI,
    #
    # Translation datasets
    #
    't5-en2de':                     textattack.datasets.translation.NewsTest2013EnglishToGerman,
}

HUGGINGFACE_DATASET_BY_MODEL = {
    #
    # bert-base-uncased
    #
    'bert-base-uncased-cola':       ('textattack/bert-base-uncased-CoLA',  ('glue', 'cola',  'validation')),
    'bert-base-uncased-mnli':       ('textattack/bert-base-uncased-MNLI',  ('glue', 'mnli',  'validation_matched', [1, 2, 0])),
    'bert-base-uncased-mrpc':       ('textattack/bert-base-uncased-MRPC',  ('glue', 'mrpc',  'validation')),
    'bert-base-uncased-qnli':       ('textattack/bert-base-uncased-QNLI',  ('glue', 'qnli',  'validation')),
    'bert-base-uncased-qqp':        ('textattack/bert-base-uncased-QQP',   ('glue', 'qqp',   'validation')),
    'bert-base-uncased-rte':        ('textattack/bert-base-uncased-RTE',   ('glue', 'rte',   'validation')),
    'bert-base-uncased-sst2':       ('textattack/bert-base-uncased-SST-2', ('glue', 'sst2', 'validation')), 
    'bert-base-uncased-stsb':       ('textattack/bert-base-uncased-STS-B', ('glue', 'stsb', 'validation', None, 5.0)), 
    'bert-base-uncased-wnli':       ('textattack/bert-base-uncased-WNLI',  ('glue', 'wnli',  'validation')),
    #
    # distilbert-base-cased
    #
    'distilbert-base-cased-cola':   ('textattack/distilbert-base-cased-CoLA',   ('glue', 'cola',  'validation')),
    'distilbert-base-cased-mrpc':   ('textattack/distilbert-base-cased-MRPC',   ('glue', 'mrpc',  'validation')),
    'distilbert-base-cased-qqp':    ('textattack/distilbert-base-cased-QQP',    ('glue', 'qqp',   'validation')),
    'distilbert-base-cased-sst2':   ('textattack/distilbert-base-cased-SST-2',  ('glue', 'sst2', 'validation')),
    'distilbert-base-cased-stsb':   ('textattack/distilbert-base-cased-STS-B',  ('glue', 'stsb', 'validation', None, 5.0)),
    #
    # distilbert-base-uncased
    #
    'distilbert-base-uncased-mnli':  ('textattack/distilbert-base-uncased-MNLI',  ('glue', 'mnli',  'validation_matched', [1, 2, 0])),
    'distilbert-base-uncased-mrpc':  ('textattack/distilbert-base-uncased-MRPC',  ('glue', 'mrpc',  'validation')),
    'distilbert-base-uncased-qnli':  ('textattack/distilbert-base-uncased-QNLI',  ('glue', 'qnli',  'validation')),
    'distilbert-base-uncased-qqp':   ('textattack/distilbert-base-uncased-QQP',   ('glue', 'qqp',   'validation')),
    'distilbert-base-uncased-rte':   ('textattack/distilbert-base-uncased-RTE',   ('glue', 'rte',   'validation')),
    'distilbert-base-uncased-sst2':  ('textattack/distilbert-base-uncased-SST-2', ('glue', 'sst2',  'validation')),
    'distilbert-base-uncased-stsb':  ('textattack/distilbert-base-uncased-STS-B', ('glue', 'stsb',  'validation', None, 5.0)),
    'distilbert-base-uncased-wnli':  ('textattack/distilbert-base-uncased-WNLI',  ('glue', 'wnli',  'validation')),
    #
    # roberta-base (RoBERTa is cased by default)
    #
    'roberta-base-cola':             ('textattack/roberta-base-CoLA',  ('glue', 'cola',  'validation')),
    'roberta-base-mrpc':             ('textattack/roberta-base-MRPC',  ('glue', 'mrpc',  'validation')),
    'roberta-base-qnli':             ('textattack/roberta-base-QNLI',  ('glue', 'qnli',  'validation')),
    'roberta-base-rte':              ('textattack/roberta-base-RTE',   ('glue', 'rte',   'validation')),
    'roberta-base-sst2':             ('textattack/roberta-base-SST-2', ('glue', 'sst2', 'validation')),
    'roberta-base-stsb':             ('textattack/roberta-base-STS-B', ('glue', 'stsb', 'validation', None, 5.0)),
    'roberta-base-wnli':             ('textattack/roberta-base-WNLI',  ('glue', 'wnli',  'validation')),
    
}

BLACK_BOX_TRANSFORMATION_CLASS_NAMES = {
    'word-swap-embedding':                  'textattack.transformations.WordSwapEmbedding',
    'word-swap-homoglyph':                  'textattack.transformations.WordSwapHomoglyphSwap',
    'word-swap-neighboring-char-swap':      'textattack.transformations.WordSwapNeighboringCharacterSwap',
    'word-swap-random-char-deletion':       'textattack.transformations.WordSwapRandomCharacterDeletion',
    'word-swap-random-char-insertion':      'textattack.transformations.WordSwapRandomCharacterInsertion',
    'word-swap-random-char-substitution':   'textattack.transformations.WordSwapRandomCharacterSubstitution',
    'word-swap-wordnet':                    'textattack.transformations.WordSwapWordNet',
}

WHITE_BOX_TRANSFORMATION_CLASS_NAMES = {
    'word-swap-gradient':                   'textattack.transformations.WordSwapGradientBased'
}

CONSTRAINT_CLASS_NAMES = {
    #
    # Semantics constraints
    #
    'embedding':        'textattack.constraints.semantics.WordEmbeddingDistance',
    'bert':             'textattack.constraints.semantics.sentence_encoders.BERT',
    'infer-sent':       'textattack.constraints.semantics.sentence_encoders.InferSent',
    'thought-vector':   'textattack.constraints.semantics.sentence_encoders.ThoughtVector',
    'use':              'textattack.constraints.semantics.sentence_encoders.UniversalSentenceEncoder',
    #
    # Grammaticality constraints
    #
    'lang-tool':        'textattack.constraints.grammaticality.LanguageTool', 
    'part-of-speech':   'textattack.constraints.grammaticality.PartOfSpeech', 
    'goog-lm':          'textattack.constraints.grammaticality.language_models.GoogleLanguageModel',
    'gpt2':             'textattack.constraints.grammaticality.language_models.GPT2',
    #
    # Overlap constraints
    #
    'bleu':                 'textattack.constraints.overlap.BLEU', 
    'chrf':                 'textattack.constraints.overlap.chrF', 
    'edit-distance':        'textattack.constraints.overlap.LevenshteinEditDistance',
    'meteor':               'textattack.constraints.overlap.METEOR',
    'max-words-perturbed':  'textattack.constraints.overlap.MaxWordsPerturbed',
    #
    # Pre-transformation constraints
    #
    'repeat':           'textattack.constraints.pre_transformation.RepeatModification',
    'stopword':         'textattack.constraints.pre_transformation.StopwordModification',
}

SEARCH_CLASS_NAMES = {
    'beam-search':      'textattack.search_methods.BeamSearch',
    'greedy':           'textattack.search_methods.GreedySearch',
    'ga-word':          'textattack.search_methods.GeneticAlgorithm',
    'greedy-word-wir':  'textattack.search_methods.GreedyWordSwapWIR',
}

GOAL_FUNCTION_CLASS_NAMES = {
    'non-overlapping-output':     'textattack.goal_functions.NonOverlappingOutput',
    'targeted-classification':    'textattack.goal_functions.TargetedClassification',
    'untargeted-classification':  'textattack.goal_functions.UntargetedClassification',
}

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

def get_args():
    # Parser for regular arguments
    parser = argparse.ArgumentParser(
        description='A commandline parser for TextAttack', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    transformation_names = set(BLACK_BOX_TRANSFORMATION_CLASS_NAMES.keys()) | set(WHITE_BOX_TRANSFORMATION_CLASS_NAMES.keys())
    parser.add_argument('--transformation', type=str, required=False,
        default='word-swap-embedding', choices=transformation_names,
        help='The transformation to apply. Usage: "--transformation {transformation}:{arg_1}={value_1},{arg_3}={value_3}. Choices: ' + str(transformation_names))
    
    model_group = parser.add_mutually_exclusive_group()
    
    model_names = list(TEXTATTACK_MODEL_CLASS_NAMES.keys()) + list(HUGGINGFACE_DATASET_BY_MODEL.keys())
    model_group.add_argument('--model', type=str, required=False, default='bert-base-uncased-yelp-sentiment',
        choices=model_names, help='The pre-trained model to attack.')
    model_group.add_argument('--model-from-file', type=str, required=False,
        help='File of model and tokenizer to import.')     
    model_group.add_argument('--model-from-huggingface', type=str, required=False,
        help='huggingface.co ID of pre-trained model to load')
        
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument('--dataset-from-nlp', type=str, required=False, default=None,
        help='Dataset to load from `nlp` repository.')
    dataset_group.add_argument('--dataset-from-file', type=str, required=False, default=None,
        help='Dataset to load from a file.')
    
    parser.add_argument('--constraints', type=str, required=False, nargs='*',
        default=['repeat', 'stopword'],
        help='Constraints to add to the attack. Usage: "--constraints {constraint}:{arg_1}={value_1},{arg_3}={value_3}". Choices: ' + str(CONSTRAINT_CLASS_NAMES.keys()))
    
    parser.add_argument('--out-dir', type=str, required=False, default=None,
        help='A directory to output results to.')
    
    parser.add_argument('--enable-visdom', action='store_true',
        help='Enable logging to visdom.')
    
    parser.add_argument('--enable-wandb', action='store_true',
        help='Enable logging to Weights & Biases.')
    
    parser.add_argument('--disable-stdout', action='store_true',
        help='Disable logging to stdout')
   
    parser.add_argument('--enable-csv', nargs='?', default=None, const='fancy', type=str,
        help='Enable logging to csv. Use --enable-csv plain to remove [[]] around words.')

    parser.add_argument('--num-examples', '-n', type=int, required=False, 
        default='5', help='The number of examples to process.')
    
    parser.add_argument('--num-examples-offset', '-o', type=int, required=False, 
        default=0, help='The offset to start at in the dataset.')

    parser.add_argument('--shuffle', action='store_true', required=False, 
        default=False, help='Randomly shuffle the data before attacking')
    
    parser.add_argument('--interactive', action='store_true', default=False,
        help='Whether to run attacks interactively.')
    
    parser.add_argument('--attack-n', action='store_true', default=False,
        help='Whether to run attack until `n` examples have been attacked (not skipped).')
    
    parser.add_argument('--parallel', action='store_true', default=False,
        help='Run attack using multiple GPUs.')

    goal_function_choices = ', '.join(GOAL_FUNCTION_CLASS_NAMES.keys())
    parser.add_argument('--goal-function', '-g', default='untargeted-classification',
        help=f'The goal function to use. choices: {goal_function_choices}')
   
    def str_to_int(s): return sum((ord(c) for c in s))
    parser.add_argument('--random-seed', default=str_to_int('TEXTATTACK'))

    parser.add_argument('--checkpoint-dir', required=False, type=str, default=default_checkpoint_dir(),
        help='The directory to save checkpoint files.')

    parser.add_argument('--checkpoint-interval', required=False, type=int, 
        help='If set, checkpoint will be saved after attacking every N examples. If not set, no checkpoints will be saved.')

    parser.add_argument('--query-budget', '-q', type=int, default=float('inf'),
        help='The maximum number of model queries allowed per example attacked.')

    attack_group = parser.add_mutually_exclusive_group(required=False)
    search_choices = ', '.join(SEARCH_CLASS_NAMES.keys())
    attack_group.add_argument('--search', '--search-method', '-s', type=str, 
        required=False, default='greedy-word-wir', 
        help=f'The search method to use. choices: {search_choices}')
    attack_group.add_argument('--recipe', '--attack-recipe', '-r', type=str, required=False, default=None,
        help='full attack recipe (overrides provided goal function, transformation & constraints)',
        choices=RECIPE_NAMES.keys())
    attack_group.add_argument('--attack-from-file', type=str, required=False, default=None,
        help='attack to load from file (overrides provided goal function, transformation & constraints)',
        )

    # Parser for parsing args for resume
    resume_parser = argparse.ArgumentParser(
        description='A commandline parser for TextAttack', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    resume_parser.add_argument('--checkpoint-file', '-f', type=str, required=True, 
        help='Path of checkpoint file to resume attack from. If "latest" (or "{directory path}/latest") is entered,'\
        'recover latest checkpoint from either current path or specified directory.')

    resume_parser.add_argument('--checkpoint-dir', '-d', required=False, type=str, default=None,
        help='The directory to save checkpoint files. If not set, use directory from recovered arguments.')

    resume_parser.add_argument('--checkpoint-interval', '-i', required=False, type=int, 
        help='If set, checkpoint will be saved after attacking every N examples. If not set, no checkpoints will be saved.')

    resume_parser.add_argument('--parallel', action='store_true', default=False,
        help='Run attack using multiple GPUs.')
    
    # Resume attack from checkpoint.
    if sys.argv[1:] and sys.argv[1].lower() == 'resume':
        args = resume_parser.parse_args(sys.argv[2:])
        setattr(args, 'checkpoint_resume', True)
    else:
        command_line_args = None if sys.argv[1:] else ['-h'] # Default to help with empty arguments.
        args = parser.parse_args(command_line_args)
        setattr(args, 'checkpoint_resume', False)

        if args.checkpoint_interval and args.shuffle:
            # Not allowed b/c we cannot recover order of shuffled data
            raise ValueError('Cannot use `--checkpoint-interval` with `--shuffle=True`')
        
        set_seed(args.random_seed)
    
    # Shortcuts for huggingface models using --model.
    if args.model in HUGGINGFACE_DATASET_BY_MODEL:
        args.model_from_huggingface, args.dataset_from_nlp = HUGGINGFACE_DATASET_BY_MODEL[args.model]
        args.model = None
    
    return args

def parse_transformation_from_args(args, model):
    # Transformations
    transformation_name = args.transformation
    if ':' in transformation_name:
        transformation_name, params = transformation_name.split(':')
        
        if transformation_name in WHITE_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(f'{WHITE_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}(model, {params})')
        elif transformation_name in BLACK_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(f'{BLACK_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}({params})')
        else:
            raise ValueError(f'Error: unsupported transformation {transformation_name}')
    else:
        if transformation_name in WHITE_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(f'{WHITE_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}(model)')
        elif transformation_name in BLACK_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(f'{BLACK_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}()')
        else:
            raise ValueError(f'Error: unsupported transformation {transformation_name}')
    return transformation

def parse_goal_function_from_args(args, model):
    # Goal Functions
    goal_function = args.goal_function
    if ':' in goal_function:
        goal_function_name, params = goal_function.split(':')
        if goal_function_name not in GOAL_FUNCTION_CLASS_NAMES:
            raise ValueError(f'Error: unsupported goal_function {goal_function_name}')
        goal_function = eval(f'{GOAL_FUNCTION_CLASS_NAMES[goal_function_name]}(model, {params})')
    elif goal_function in GOAL_FUNCTION_CLASS_NAMES:
        goal_function = eval(f'{GOAL_FUNCTION_CLASS_NAMES[goal_function]}(model)')
    else:
        raise ValueError(f'Error: unsupported goal_function {goal_function}')
    goal_function.query_budget = args.query_budget
    return goal_function

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

def parse_attack_from_args(args):
    model = parse_model_from_args(args)
    if args.recipe:
        if ':' in args.recipe:
            recipe_name, params = args.recipe.split(':')
            if recipe_name not in RECIPE_NAMES:
                raise ValueError(f'Error: unsupported recipe {recipe_name}')
            recipe = eval(f'{RECIPE_NAMES[recipe_name]}(model, {params})')
        elif args.recipe in RECIPE_NAMES:
            recipe = eval(f'{RECIPE_NAMES[args.recipe]}(model)')
        else:
            raise ValueError(f'Invalid recipe {args.recipe}')
        recipe.goal_function.query_budget = args.query_budget
        return recipe
    elif args.attack_from_file:
        if ':' in args.attack_from_file:
            attack_file, attack_name = args.attack_from_file.split(':')
        else:
            attack_file, attack_name = args.attack_from_file, 'attack'
        attack_file = attack_file.replace('.py', '').replace('/', '.')
        attack_module = importlib.import_module(attack_file)
        attack_func = getattr(attack_module, attack_name)
        return attack_func(model)
    else:
        goal_function = parse_goal_function_from_args(args, model)
        transformation = parse_transformation_from_args(args, model)
        constraints = parse_constraints_from_args(args)
        if ':' in args.search:
            search_name, params = args.search.split(':')
            if search_name not in SEARCH_CLASS_NAMES:
                raise ValueError(f'Error: unsupported search {search_name}')
            search_method = eval(f'{SEARCH_CLASS_NAMES[search_name]}({params})')
        elif args.search in SEARCH_CLASS_NAMES:
            search_method = eval(f'{SEARCH_CLASS_NAMES[args.search]}()')
        else:
            raise ValueError(f'Error: unsupported attack {args.search}')
    return textattack.shared.Attack(goal_function, constraints, transformation, search_method)

def parse_model_from_args(args):
    if args.model_from_file:
        colored_model_name = textattack.shared.utils.color_text(args.model_from_file, color='blue', method='ansi')
        textattack.shared.logger.info(f'Loading model and tokenizer from file: {colored_model_name}')
        if ':' in args.model_from_file:
            model_file, model_name, tokenizer_name = args.model_from_file.split(':')
        else:
            model_file, model_name, tokenizer_name = args.model_from_file, 'model', 'tokenizer'
        try:
            model_file = args.model_from_file.replace('.py', '').replace('/', '.')
            model_module = importlib.import_module(model_file)
        except:
            raise ValueError(f'Failed to import model or tokenizer from file {args.model_from_file}')
        try:
            model = getattr(model_module, model_name)
        except AttributeError:
            raise AttributeError(f'``{model_name}`` not found in module {args.model_from_file}')
        try:
            tokenizer = getattr(model_module, tokenizer_name)
        except AttributeError:
            raise AttributeError(f'``{tokenizer_name}`` not found in module {args.model_from_file}')
        model = model.to(textattack.shared.utils.device)
        setattr(model, 'tokenizer', tokenizer)
    elif args.model_from_huggingface:
        import transformers
        if ':' in args.model_from_huggingface:
            model_class, model_name = args.model_from_huggingface.split(':')
            model_class = eval(f'transformers.{model_class}')
        else:
            model_class, model_name = transformers.AutoModelForSequenceClassification, args.model_from_huggingface
        colored_model_name = textattack.shared.utils.color_text(model_name, color='blue', method='ansi')
        textattack.shared.logger.info(f'Loading pre-trained model from HuggingFace model repository: {colored_model_name}')
        model = model_class.from_pretrained(model_name)
        model = model.to(textattack.shared.utils.device)
        try:
            tokenizer = textattack.tokenizers.AutoTokenizer(args.model_from_huggingface)
        except OSError:
            textattack.shared.logger.warn(f'AutoTokenizer {args.model_from_huggingface} not found. Defaulting to `bert-base-uncased`')
            tokenizer = textattack.tokenizers.AutoTokenizer('bert-base-uncased')
        setattr(model, 'tokenizer', tokenizer)
    else:
        if ':' in args.model:
            model_name, params = args.model.split(':')
            colored_model_name = textattack.shared.utils.color_text(model_name, color='blue', method='ansi')
            textattack.shared.logger.info(f'Loading pre-trained TextAttack model: {colored_model_name}')
            if model_name not in TEXTATTACK_MODEL_CLASS_NAMES:
                raise ValueError(f'Error: unsupported model {model_name}')
            model = eval(f'{TEXTATTACK_MODEL_CLASS_NAMES[model_name]}({params})')
        elif args.model in TEXTATTACK_MODEL_CLASS_NAMES:
            colored_model_name = textattack.shared.utils.color_text(args.model, color='blue', method='ansi')
            textattack.shared.logger.info(f'Loading pre-trained TextAttack model: {colored_model_name}')
            model = eval(f'{TEXTATTACK_MODEL_CLASS_NAMES[args.model]}()')
        else: 
            raise ValueError(f'Error: unsupported model {args.model}')
    return model

def parse_dataset_from_args(args):
    if args.dataset_from_file:
        textattack.shared.logger.info(f'Loading model and tokenizer from file: {args.model_from_file}')
        if ':' in args.dataset_from_file:
            dataset_file, dataset_name = args.dataset_from_file.split(':')
        else:
            dataset_file, dataset_name = args.dataset_from_file, 'dataset'
        try:
            dataset_file = dataset_file.replace('.py', '').replace('/', '.')
            dataset_module = importlib.import_module(dataset_file)
        except:
            raise ValueError(f'Failed to import dataset from file {args.dataset_from_file}')
        try:
            dataset = getattr(dataset_module, dataset_name)
        except AttributeError:
            raise AttributeError(f'``dataset`` not found in module {args.dataset_from_file}')
    elif args.dataset_from_nlp:
        dataset_args = args.dataset_from_nlp
        if ':' in dataset_args:
            dataset_args = dataset_args.split(':')
        dataset = textattack.datasets.HuggingFaceNLPDataset(*dataset_args, shuffle=args.shuffle)
    else:
        if not args.model:
            raise ValueError('Must supply pretrained model or dataset')
        elif args.model in DATASET_BY_MODEL:
            dataset = DATASET_BY_MODEL[args.model](offset=args.num_examples_offset)
        else:
            raise ValueError(f'Error: unsupported model {args.model}')
    return dataset

def parse_logger_from_args(args):
    # Create logger
    attack_log_manager = textattack.loggers.AttackLogManager()
    # Set default output directory to `textattack/outputs`.
    if not args.out_dir:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        outputs_dir = os.path.join(current_dir, os.pardir, os.pardir, os.pardir, 'outputs')
        args.out_dir = os.path.normpath(outputs_dir)
        
    # Output file.
    out_time = int(time.time()*1000) # Output file
    outfile_name = 'attack-{}.txt'.format(out_time)
    attack_log_manager.add_output_file(os.path.join(args.out_dir, outfile_name))
        
    # CSV
    if args.enable_csv:
        outfile_name = 'attack-{}.csv'.format(out_time)
        color_method = None if args.enable_csv == 'plain' else 'file'
        csv_path = os.path.join(args.out_dir, outfile_name)
        attack_log_manager.add_output_csv(csv_path, color_method)
        print('Logging to CSV at path {}.'.format(csv_path))

    # Visdom
    if args.enable_visdom:
        attack_log_manager.enable_visdom()
        
    # Weights & Biases
    if args.enable_wandb:
        attack_log_manager.enable_wandb()

    # Stdout
    if not args.disable_stdout:
        attack_log_manager.enable_stdout()
    return attack_log_manager

def parse_checkpoint_from_args(args):
    file_name = os.path.basename(args.checkpoint_file)   
    if file_name.lower() == 'latest':
        dir_path = os.path.dirname(args.checkpoint_file)
        chkpt_file_names = [f for f in os.listdir(dir_path) if f.endswith('.ta.chkpt')]
        assert chkpt_file_names, "Checkpoint directory is empty"
        timestamps = [int(f.replace('.ta.chkpt', '')) for f in chkpt_file_names]
        latest_file = str(max(timestamps)) + '.ta.chkpt'
        checkpoint_path = os.path.join(dir_path, latest_file)
    else:
        checkpoint_path = args.checkpoint_file
    
    checkpoint = textattack.shared.Checkpoint.load(checkpoint_path)
    set_seed(checkpoint.args.random_seed)

    return checkpoint

def default_checkpoint_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoints_dir = os.path.join(current_dir, os.pardir, os.pardir, os.pardir, 'checkpoints')
    return os.path.normpath(checkpoints_dir)

def merge_checkpoint_args(saved_args, cmdline_args):
    """ Merge previously saved arguments for checkpoint and newly entered arguments """
    args = copy.deepcopy(saved_args)
    # Newly entered arguments take precedence
    args.checkpoint_resume = cmdline_args.checkpoint_resume
    args.parallel =  cmdline_args.parallel
    # If set, replace
    if cmdline_args.checkpoint_dir:
        args.checkpoint_dir = cmdline_args.checkpoint_dir
    if cmdline_args.checkpoint_interval:
        args.checkpoint_interval = cmdline_args.checkpoint_interval
    
    return args
