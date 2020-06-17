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

from .attack_args import *

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
    
    model_names = list(TEXTATTACK_MODEL_CLASS_NAMES.keys()) + list(HUGGINGFACE_DATASET_BY_MODEL.keys()) + list(TEXTATTACK_DATASET_BY_MODEL.keys())
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
        elif args.model in TEXTATTACK_DATASET_BY_MODEL:
            colored_model_name = textattack.shared.utils.color_text(args.model, color='blue', method='ansi')
            model_path, args.dataset_from_nlp = TEXTATTACK_DATASET_BY_MODEL[args.model]
            if args.model.startswith('lstm'):
                textattack.shared.logger.info(f'Loading pre-trained TextAttack LSTM: {colored_model_name}')
                model = textattack.models.helpers.LSTMForClassification(model_path=model_path)
            elif args.model.startswith('cnn'):
                textattack.shared.logger.info(f'Loading pre-trained TextAttack CNN: {colored_model_name}')
                model = textattack.models.helpers.WordCNNForClassification(model_path=model_path)
            else:
                raise ValueError(f'Unknown TextAttack pretrained model {args.model}')
        else: 
            raise ValueError(f'Error: unsupported model {args.model}')
    return model

def parse_dataset_from_args(args):
    args.dataset_from_nlp = ('glue', 'sst2', 'validation')
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
