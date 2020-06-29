import argparse
import copy
import importlib
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

import textattack

from .attack_args import *


def add_model_args(parser):
    """ Adds model-related arguments to an argparser. This is useful because we
        want to load pretrained models using multiple different parsers that
        share these, but not all, arguments.
    """
    model_group = parser.add_mutually_exclusive_group()

    model_names = list(HUGGINGFACE_DATASET_BY_MODEL.keys()) + list(
        TEXTATTACK_DATASET_BY_MODEL.keys()
    )
    model_group.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help='Name of or path to a pre-trained model to attack. Usage: "--model {model}:{arg_1}={value_1},{arg_3}={value_3},...". Choices: '
        + str(model_names),
    )
    model_group.add_argument(
        "--model-from-file",
        type=str,
        required=False,
        help="File of model and tokenizer to import.",
    )
    model_group.add_argument(
        "--model-from-huggingface",
        type=str,
        required=False,
        help="huggingface.co ID of pre-trained model to load",
    )


def add_dataset_args(parser):
    """ Adds dataset-related arguments to an argparser. This is useful because we
        want to load pretrained models using multiple different parsers that
        share these, but not all, arguments.
    """
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset-from-nlp",
        type=str,
        required=False,
        default=None,
        help="Dataset to load from `nlp` repository.",
    )
    dataset_group.add_argument(
        "--dataset-from-file",
        type=str,
        required=False,
        default=None,
        help="Dataset to load from a file.",
    )
    dataset_group.add_argument(
        "--shuffle",
        action="store_true",
        required=False,
        default=False,
        help="Randomly shuffle the data before attacking",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        required=False,
        default="5",
        help="The number of examples to process.",
    )

    parser.add_argument(
        "--num-examples-offset",
        "-o",
        type=int,
        required=False,
        default=0,
        help="The offset to start at in the dataset.",
    )


def load_module_from_file(file_path):
    """ Uses ``importlib`` to dynamically open a file and load an object from
        it. 
    """
    temp_module_name = f"temp_{time.time()}"
    colored_file_path = textattack.shared.utils.color_text(
        file_path, color="blue", method="ansi"
    )
    textattack.shared.logger.info(f"Loading module from `{colored_file_path}`.")
    spec = importlib.util.spec_from_file_location(temp_module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_transformation_from_args(args, model):
    # Transformations
    transformation_name = args.transformation
    if ":" in transformation_name:
        transformation_name, params = transformation_name.split(":")

        if transformation_name in WHITE_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{WHITE_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}(model, {params})"
            )
        elif transformation_name in BLACK_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{BLACK_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}({params})"
            )
        else:
            raise ValueError(f"Error: unsupported transformation {transformation_name}")
    else:
        if transformation_name in WHITE_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{WHITE_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}(model)"
            )
        elif transformation_name in BLACK_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{BLACK_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}()"
            )
        else:
            raise ValueError(f"Error: unsupported transformation {transformation_name}")
    return transformation


def parse_goal_function_from_args(args, model):
    # Goal Functions
    goal_function = args.goal_function
    if ":" in goal_function:
        goal_function_name, params = goal_function.split(":")
        if goal_function_name not in GOAL_FUNCTION_CLASS_NAMES:
            raise ValueError(f"Error: unsupported goal_function {goal_function_name}")
        goal_function = eval(
            f"{GOAL_FUNCTION_CLASS_NAMES[goal_function_name]}(model, {params})"
        )
    elif goal_function in GOAL_FUNCTION_CLASS_NAMES:
        goal_function = eval(f"{GOAL_FUNCTION_CLASS_NAMES[goal_function]}(model)")
    else:
        raise ValueError(f"Error: unsupported goal_function {goal_function}")
    goal_function.query_budget = args.query_budget
    goal_function.model_batch_size = args.model_batch_size
    goal_function.model_cache_size = args.model_cache_size
    return goal_function


def parse_constraints_from_args(args):
    # Constraints
    if not args.constraints:
        return []

    _constraints = []
    for constraint in args.constraints:
        if ":" in constraint:
            constraint_name, params = constraint.split(":")
            if constraint_name not in CONSTRAINT_CLASS_NAMES:
                raise ValueError(f"Error: unsupported constraint {constraint_name}")
            _constraints.append(
                eval(f"{CONSTRAINT_CLASS_NAMES[constraint_name]}({params})")
            )
        elif constraint in CONSTRAINT_CLASS_NAMES:
            _constraints.append(eval(f"{CONSTRAINT_CLASS_NAMES[constraint]}()"))
        else:
            raise ValueError(f"Error: unsupported constraint {constraint}")

    return _constraints


def parse_attack_from_args(args):
    model = parse_model_from_args(args)
    if args.recipe:
        if ":" in args.recipe:
            recipe_name, params = args.recipe.split(":")
            if recipe_name not in ATTACK_RECIPE_NAMES:
                raise ValueError(f"Error: unsupported recipe {recipe_name}")
            recipe = eval(f"{ATTACK_RECIPE_NAMES[recipe_name]}(model, {params})")
        elif args.recipe in ATTACK_RECIPE_NAMES:
            recipe = eval(f"{ATTACK_RECIPE_NAMES[args.recipe]}(model)")
        else:
            raise ValueError(f"Invalid recipe {args.recipe}")
        recipe.goal_function.query_budget = args.query_budget
        recipe.goal_function.model_batch_size = args.model_batch_size
        recipe.goal_function.model_cache_size = args.model_cache_size
        recipe.constraint_cache_size = args.constraint_cache_size
        return recipe
    elif args.attack_from_file:
        if ":" in args.attack_from_file:
            attack_file, attack_name = args.attack_from_file.split(":")
        else:
            attack_file, attack_name = args.attack_from_file, "attack"
        attack_module = load_module_from_file(attack_file)
        if not hasattr(attack_module, attack_name):
            raise ValueError(
                f"Loaded `{attack_file}` but could not find `{attack_name}`."
            )
        attack_func = getattr(attack_module, attack_name)
        return attack_func(model)
    else:
        goal_function = parse_goal_function_from_args(args, model)
        transformation = parse_transformation_from_args(args, model)
        constraints = parse_constraints_from_args(args)
        if ":" in args.search:
            search_name, params = args.search.split(":")
            if search_name not in SEARCH_METHOD_CLASS_NAMES:
                raise ValueError(f"Error: unsupported search {search_name}")
            search_method = eval(f"{SEARCH_METHOD_CLASS_NAMES[search_name]}({params})")
        elif args.search in SEARCH_METHOD_CLASS_NAMES:
            search_method = eval(f"{SEARCH_METHOD_CLASS_NAMES[args.search]}()")
        else:
            raise ValueError(f"Error: unsupported attack {args.search}")
    return textattack.shared.Attack(
        goal_function,
        constraints,
        transformation,
        search_method,
        constraint_cache_size=args.constraint_cache_size,
    )


def parse_model_from_args(args):
    if args.model_from_file:
        colored_model_name = textattack.shared.utils.color_text(
            args.model_from_file, color="blue", method="ansi"
        )
        textattack.shared.logger.info(
            f"Loading model and tokenizer from file: {colored_model_name}"
        )
        if ":" in args.model_from_file:
            model_file, model_name, tokenizer_name = args.model_from_file.split(":")
        else:
            model_file, model_name, tokenizer_name = (
                args.model_from_file,
                "model",
                "tokenizer",
            )
        try:
            model_module = load_module_from_file(args.model_from_file)
        except:
            raise ValueError(f"Failed to import file {args.model_from_file}")
        try:
            model = getattr(model_module, model_name)
        except AttributeError:
            raise AttributeError(
                f"``{model_name}`` not found in module {args.model_from_file}"
            )
        try:
            tokenizer = getattr(model_module, tokenizer_name)
        except AttributeError:
            raise AttributeError(
                f"``{tokenizer_name}`` not found in module {args.model_from_file}"
            )
        model = model.to(textattack.shared.utils.device)
        setattr(model, "tokenizer", tokenizer)
    elif (args.model in HUGGINGFACE_DATASET_BY_MODEL) or args.model_from_huggingface:
        import transformers

        model_name = (
            HUGGINGFACE_DATASET_BY_MODEL[args.model][0]
            if (args.model in HUGGINGFACE_DATASET_BY_MODEL)
            else args.model_from_huggingface
        )

        if ":" in model_name:
            model_class, model_name = model_name
            model_class = eval(f"transformers.{model_class}")
        else:
            model_class, model_name = (
                transformers.AutoModelForSequenceClassification,
                model_name,
            )
        colored_model_name = textattack.shared.utils.color_text(
            model_name, color="blue", method="ansi"
        )
        textattack.shared.logger.info(
            f"Loading pre-trained model from HuggingFace model repository: {colored_model_name}"
        )
        model = model_class.from_pretrained(model_name)
        model = model.to(textattack.shared.utils.device)
        try:
            tokenizer = textattack.models.tokenizers.AutoTokenizer(model_name)
        except OSError:
            textattack.shared.logger.warn(
                f"AutoTokenizer {args.model_from_huggingface} not found. Defaulting to `bert-base-uncased`"
            )
            tokenizer = textattack.models.tokenizers.AutoTokenizer("bert-base-uncased")
        setattr(model, "tokenizer", tokenizer)
    else:
        if args.model in TEXTATTACK_DATASET_BY_MODEL:
            model_path, _ = TEXTATTACK_DATASET_BY_MODEL[args.model]
            model = textattack.shared.utils.load_textattack_model_from_path(
                args.model, model_path
            )
        elif args.model and os.path.exists(args.model):
            # If `args.model` is a path/directory, let's assume it was a model
            # trained with textattack, and try and load it.
            model_args_json_path = os.path.join(args.model, "train_args.json")
            if not os.path.exists(model_args_json_path):
                raise FileNotFoundError(
                    f"Tried to load model from path {args.model} - could not find train_args.json."
                )
            model_train_args = json.loads(open(model_args_json_path).read())
            model_train_args["model"] = args.model
            num_labels = model_train_args["num_labels"]
            from textattack.commands.train_model.train_args_helpers import (
                model_from_args,
            )

            model = model_from_args(argparse.Namespace(**model_train_args), num_labels)
        else:
            raise ValueError(f"Error: unsupported TextAttack model {args.model}")
    return model


def parse_dataset_from_args(args):
    # Automatically detect dataset for huggingface & textattack models.
    # This allows us to use the --model shortcut without specifying a dataset.
    if args.model in HUGGINGFACE_DATASET_BY_MODEL:
        _, args.dataset_from_nlp = HUGGINGFACE_DATASET_BY_MODEL[args.model]
    elif args.model in TEXTATTACK_DATASET_BY_MODEL:
        _, args.dataset_from_nlp = TEXTATTACK_DATASET_BY_MODEL[args.model]
    # Automatically detect dataset for models trained with textattack.
    elif args.model and os.path.exists(args.model):
        model_args_json_path = os.path.join(args.model, "train_args.json")
        if not os.path.exists(model_args_json_path):
            raise FileNotFoundError(
                f"Tried to load model from path {args.model} - could not find train_args.json."
            )
        model_train_args = json.loads(open(model_args_json_path).read())
        try:
            args.dataset_from_nlp = (
                model_train_args["dataset"],
                None,
                model_train_args["dataset_dev_split"],
            )
        except KeyError:
            raise KeyError(
                f"Tried to load model from path {args.model} but can't initialize dataset from train_args.json."
            )

    # Get dataset from args.
    if args.dataset_from_file:
        textattack.shared.logger.info(
            f"Loading model and tokenizer from file: {args.model_from_file}"
        )
        if ":" in args.dataset_from_file:
            dataset_file, dataset_name = args.dataset_from_file.split(":")
        else:
            dataset_file, dataset_name = args.dataset_from_file, "dataset"
        try:
            dataset_module = load_module_from_file(dataset_file)
        except:
            raise ValueError(
                f"Failed to import dataset from file {args.dataset_from_file}"
            )
        try:
            dataset = getattr(dataset_module, dataset_name)
        except AttributeError:
            raise AttributeError(
                f"``dataset`` not found in module {args.dataset_from_file}"
            )
    elif args.dataset_from_nlp:
        dataset_args = args.dataset_from_nlp
        if isinstance(dataset_args, str):
            if ":" in dataset_args:
                dataset_args = dataset_args.split(":")
            else:
                dataset_args = (dataset_args,)
        dataset = textattack.datasets.HuggingFaceNLPDataset(
            *dataset_args, shuffle=args.shuffle
        )
        dataset.examples = dataset.examples[args.num_examples_offset :]
    else:
        raise ValueError("Must supply pretrained model or dataset")
    return dataset


def parse_logger_from_args(args):
    # Create logger
    attack_log_manager = textattack.loggers.AttackLogManager()
    out_time = int(time.time() * 1000)  # Output file
    # Set default output directory to `textattack/outputs`.
    if not args.out_dir:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        outputs_dir = os.path.join(
            current_dir, os.pardir, os.pardir, os.pardir, "outputs", "attacks"
        )
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
        args.out_dir = os.path.normpath(outputs_dir)

    # if "--log-to-file" specified in terminal command, then save it to a txt file
    if args.log_to_file:
        # Output file.
        outfile_name = "attack-{}.txt".format(out_time)
        attack_log_manager.add_output_file(os.path.join(args.out_dir, outfile_name))

    # CSV
    if args.enable_csv:
        outfile_name = "attack-{}.csv".format(out_time)
        color_method = None if args.enable_csv == "plain" else "file"
        csv_path = os.path.join(args.out_dir, outfile_name)
        attack_log_manager.add_output_csv(csv_path, color_method)
        print("Logging to CSV at path {}.".format(csv_path))

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
    if file_name.lower() == "latest":
        dir_path = os.path.dirname(args.checkpoint_file)
        chkpt_file_names = [f for f in os.listdir(dir_path) if f.endswith(".ta.chkpt")]
        assert chkpt_file_names, "Checkpoint directory is empty"
        timestamps = [int(f.replace(".ta.chkpt", "")) for f in chkpt_file_names]
        latest_file = str(max(timestamps)) + ".ta.chkpt"
        checkpoint_path = os.path.join(dir_path, latest_file)
    else:
        checkpoint_path = args.checkpoint_file

    checkpoint = textattack.shared.Checkpoint.load(checkpoint_path)

    return checkpoint


def default_checkpoint_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoints_dir = os.path.join(
        current_dir, os.pardir, os.pardir, os.pardir, "checkpoints"
    )
    return os.path.normpath(checkpoints_dir)


def merge_checkpoint_args(saved_args, cmdline_args):
    """ Merge previously saved arguments for checkpoint and newly entered arguments """
    args = copy.deepcopy(saved_args)
    # Newly entered arguments take precedence
    args.parallel = cmdline_args.parallel
    # If set, replace
    if cmdline_args.checkpoint_dir:
        args.checkpoint_dir = cmdline_args.checkpoint_dir
    if cmdline_args.checkpoint_interval:
        args.checkpoint_interval = cmdline_args.checkpoint_interval

    return args
