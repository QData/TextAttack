import argparse
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
    'alz-adjusted':     'textattack.attack_recipes.Alzantot2018Adjusted',
    'deepwordbug':      'textattack.attack_recipes.DeepWordBugGao2018',
    'hotflip':          'textattack.attack_recipes.HotFlipEbrahimi2017',
    'kuleshov':         'textattack.attack_recipes.Kuleshov2017',
    'seq2sick':         'textattack.attack_recipes.Seq2SickCheng2018BlackBox',
    'textfooler':       'textattack.attack_recipes.TextFoolerJin2019',
    'tf-adjusted':      'textattack.attack_recipes.TextFoolerJin2019Adjusted',
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
    #
    # Translation models
    #
    't5-en2fr':                 'textattack.models.translation.t5.T5EnglishToFrench',
    't5-en2de':                 'textattack.models.translation.t5.T5EnglishToGerman',
    't5-en2ro':                 'textattack.models.translation.t5.T5EnglishToRomanian',
    #
    # Summarization models
    #
    't5-summ':                  'textattack.models.summarization.T5Summarization',
}

DATASET_BY_MODEL = {
    #
    # Text classification datasets
    #
    # AG News
    'bert-ag-news':             textattack.datasets.classification.AGNews,
    'cnn-ag-news':              textattack.datasets.classification.AGNews,
    'lstm-ag-news':             textattack.datasets.classification.AGNews,
    # IMDB 
    'bert-imdb':                textattack.datasets.classification.IMDBSentiment,
    'cnn-imdb':                 textattack.datasets.classification.IMDBSentiment,
    'lstm-imdb':                textattack.datasets.classification.IMDBSentiment,
    # MR
    'bert-mr':                  textattack.datasets.classification.MovieReviewSentiment,
    'cnn-mr':                   textattack.datasets.classification.MovieReviewSentiment,
    'lstm-mr':                  textattack.datasets.classification.MovieReviewSentiment,
    # Yelp
    'bert-yelp-sentiment':      textattack.datasets.classification.YelpSentiment,
    'cnn-yelp-sentiment':       textattack.datasets.classification.YelpSentiment,
    'lstm-yelp-sentiment':      textattack.datasets.classification.YelpSentiment,
    #
    # Textual entailment datasets
    #
    'bert-mnli':                textattack.datasets.entailment.MNLI,
    'bert-snli':                textattack.datasets.entailment.SNLI,
    #
    # Translation datasets
    #
    't5-en2de':                 textattack.datasets.translation.NewsTest2013EnglishToGerman,
}

TRANSFORMATION_CLASS_NAMES = {
    'word-swap-embedding':                  'textattack.transformations.WordSwapEmbedding',
    'word-swap-homoglyph':                  'textattack.transformations.WordSwapHomoglyph',
    'word-swap-neighboring-char-swap':      'textattack.transformations.WordSwapNeighboringCharacterSwap',
    'word-swap-random-char-deletion':       'textattack.transformations.WordSwapRandomCharacterDeletion',
    'word-swap-random-char-insertion':      'textattack.transformations.WordSwapRandomCharacterInsertion',
    'word-swap-random-char-substitution':   'textattack.transformations.WordSwapRandomCharacterSubstitution',
    'word-swap-wordnet':                    'textattack.transformations.WordSwapWordNet',
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

    parser.add_argument('--transformation', type=str, required=False,
        default='word-swap-embedding', choices=TRANSFORMATION_CLASS_NAMES.keys(),
        help='The transformations to apply.')

    parser.add_argument('--model', type=str, required=False, default='bert-yelp-sentiment',
        choices=MODEL_CLASS_NAMES.keys(), help='The classification model to attack.')
    
    parser.add_argument('--constraints', type=str, required=False, nargs='*',
        default=['repeat', 'stopword'],
        help=('Constraints to add to the attack. Usage: "--constraints {constraint}:{arg_1}={value_1},{arg_3}={value_3}". Choices: ' + str(CONSTRAINT_CLASS_NAMES.keys())))
    
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
        help='A directory to save/load checkpoint files.')

    parser.add_argument('--checkpoint-interval', required=False, type=int, 
        help='Interval for saving checkpoints. If not set, no checkpoints will be saved.')
    
    attack_group = parser.add_mutually_exclusive_group(required=False)
    
    search_choices = ', '.join(SEARCH_CLASS_NAMES.keys())
    attack_group.add_argument('--search', '-s', '--search-method', type=str, 
        required=False, default='greedy-word-wir', 
        help=f'The search method to use. choices: {search_choices}')
    
    attack_group.add_argument('--recipe', '-r', type=str, required=False, default=None,
        help='full attack recipe (overrides provided goal function, transformation & constraints)',
        choices=RECIPE_NAMES.keys())

    # Parser for parsing args for resume
    resume_parser = argparse.ArgumentParser(
        description='A commandline parser for TextAttack', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    resume_parser.add_argument('--checkpoint-file', '-f', type=str, required=False, default='latest', 
        help='Name of checkpoint file to resume attack from. If "latest" is entered, recover latest checkpoint.')

    resume_parser.add_argument('--checkpoint-dir', '-d', required=False, type=str, default=default_checkpoint_dir(),
        help='A directory to save/load checkpoint files.')

    resume_parser.add_argument('--checkpoint-interval', '-i', required=False, type=int, 
        help='Interval for saving checkpoints. If not set, no checkpoints will be saved.')

    resume_parser.add_argument('--parallel', action='store_true', default=False,
        help='Run attack using multiple GPUs.')

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
    
    return args

def parse_transformation_from_args(args):
    # Transformations
    transformation = args.transformation
    if ':' in transformation:
        transformation_name, params = transformation.split(':')
        if transformation_name not in TRANSFORMATION_CLASS_NAMES:
            raise ValueError(f'Error: unsupported transformation {transformation_name}')
        transformation = eval(f'{TRANSFORMATION_CLASS_NAMES[transformation_name]}({params})')
    elif transformation in TRANSFORMATION_CLASS_NAMES:
        transformation = eval(f'{TRANSFORMATION_CLASS_NAMES[transformation]}()')
    else:
        raise ValueError(f'Error: unsupported transformation {transformation}')
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

def parse_recipe_from_args(model, args):
    if ':' in args.recipe:
        recipe_name, params = args.recipe.split(':')
        if recipe_name not in RECIPE_NAMES:
            raise ValueError(f'Error: unsupported recipe {recipe_name}')
        recipe = eval(f'{RECIPE_NAMES[recipe_name]}(model, {params})')
    elif args.recipe in RECIPE_NAMES:
        recipe = eval(f'{RECIPE_NAMES[args.recipe]}(model)')
    else:
        raise ValueError('Invalid recipe {args.recipe}')
    return recipe

def parse_goal_function_and_attack_from_args(args):
    if ':' in args.model:
        model_name, params = args.model.split(':')
        if model_name not in MODEL_CLASS_NAMES:
            raise ValueError(f'Error: unsupported model {model_name}')
        model = eval(f'{MODEL_CLASS_NAMES[model_name]}({params})')
    elif args.model in MODEL_CLASS_NAMES:
        model = eval(f'{MODEL_CLASS_NAMES[args.model]}()')
    else: 
        raise ValueError(f'Error: unsupported model {args.model}')
    if args.recipe:
        attack = parse_recipe_from_args(model, args)
        goal_function = attack.goal_function
        return goal_function, attack
    else:
        goal_function = parse_goal_function_from_args(args, model)
        transformation = parse_transformation_from_args(args)
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
    return goal_function, textattack.shared.Attack(goal_function, constraints, transformation, search_method)

def parse_logger_from_args(args):# Create logger
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
    if args.checkpoint_file.lower() == 'latest':
        chkpt_file_names = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.ta.chkpt')]
        assert chkpt_file_names, "Checkpoint directory is empty"
        timestamps = [int(f.replace('.ta.chkpt', '')) for f in chkpt_file_names]
        latest_file = str(max(timestamps)) + '.ta.chkpt'
        checkpoint_path = os.path.join(args.checkpoint_dir, latest_file)
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    
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
    args.checkpoint_dir = cmdline_args.checkpoint_dir
    # If set, we replace
    if cmdline_args.checkpoint_interval:
        args.checkpoint_interval = cmdlineargs.checkpoint_interval
    
    return args
