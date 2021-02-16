"""

TextAttack Command Helpers for Attack
------------------------------------------

"""

import argparse
import copy
import importlib
import json
import os
import time

import textattack

from .attack_args import (
    ATTACK_RECIPE_NAMES,
    BLACK_BOX_TRANSFORMATION_CLASS_NAMES,
    CONSTRAINT_CLASS_NAMES,
    GOAL_FUNCTION_CLASS_NAMES,
    HUGGINGFACE_DATASET_BY_MODEL,
    SEARCH_METHOD_CLASS_NAMES,
    TEXTATTACK_DATASET_BY_MODEL,
    WHITE_BOX_TRANSFORMATION_CLASS_NAMES,
)

# The split token allows users to optionally pass multiple arguments in a single
# parameter by separating them with the split token.
ARGS_SPLIT_TOKEN = "^"


def add_model_args(parser):
    """Adds model-related arguments to an argparser.

    This is useful because we want to load pretrained models using
    multiple different parsers that share these, but not all, arguments.
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
    """Adds dataset-related arguments to an argparser.

    This is useful because we want to load pretrained models using
    multiple different parsers that share these, but not all, arguments.
    """
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset-from-huggingface",
        type=str,
        required=False,
        default=None,
        help="Dataset to load from `datasets` repository.",
    )
    dataset_group.add_argument(
        "--dataset-from-file",
        type=str,
        required=False,
        default=None,
        help="Dataset to load from a file.",
    )
    parser.add_argument(
        "--shuffle",
        type=eval,
        required=False,
        choices=[True, False],
        default="True",
        help="Randomly shuffle the data before attacking",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        required=False,
        default="5",
        help="The number of examples to process, -1 for entire dataset",
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
    """Uses ``importlib`` to dynamically open a file and load an object from
    it."""
    temp_module_name = f"temp_{time.time()}"
    colored_file_path = textattack.shared.utils.color_text(
        file_path, color="blue", method="ansi"
    )
    textattack.shared.logger.info(f"Loading module from `{colored_file_path}`.")
    spec = importlib.util.spec_from_file_location(temp_module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_transformation_from_args(args, model_wrapper):
    transformation_name = args.transformation
    if ARGS_SPLIT_TOKEN in transformation_name:
        transformation_name, params = transformation_name.split(ARGS_SPLIT_TOKEN)

        if transformation_name in WHITE_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{WHITE_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}(model_wrapper.model, {params})"
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
                f"{WHITE_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}(model_wrapper.model)"
            )
        elif transformation_name in BLACK_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{BLACK_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}()"
            )
        else:
            raise ValueError(f"Error: unsupported transformation {transformation_name}")
    return transformation


def parse_goal_function_from_args(args, model):
    goal_function = args.goal_function
    if ARGS_SPLIT_TOKEN in goal_function:
        goal_function_name, params = goal_function.split(ARGS_SPLIT_TOKEN)
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
    if not args.constraints:
        return []

    _constraints = []
    for constraint in args.constraints:
        if ARGS_SPLIT_TOKEN in constraint:
            constraint_name, params = constraint.split(ARGS_SPLIT_TOKEN)
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
        if ARGS_SPLIT_TOKEN in args.recipe:
            recipe_name, params = args.recipe.split(ARGS_SPLIT_TOKEN)
            if recipe_name not in ATTACK_RECIPE_NAMES:
                raise ValueError(f"Error: unsupported recipe {recipe_name}")
            recipe = eval(f"{ATTACK_RECIPE_NAMES[recipe_name]}.build(model, {params})")
        elif args.recipe in ATTACK_RECIPE_NAMES:
            recipe = eval(f"{ATTACK_RECIPE_NAMES[args.recipe]}.build(model)")
        else:
            raise ValueError(f"Invalid recipe {args.recipe}")
        recipe.goal_function.query_budget = args.query_budget
        recipe.goal_function.model_batch_size = args.model_batch_size
        recipe.goal_function.model_cache_size = args.model_cache_size
        recipe.constraint_cache_size = args.constraint_cache_size
        return recipe
    elif args.attack_from_file:
        if ARGS_SPLIT_TOKEN in args.attack_from_file:
            attack_file, attack_name = args.attack_from_file.split(ARGS_SPLIT_TOKEN)
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
        if ARGS_SPLIT_TOKEN in args.search:
            search_name, params = args.search.split(ARGS_SPLIT_TOKEN)
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
        # Support loading the model from a .py file where a model wrapper
        # is instantiated.
        colored_model_name = textattack.shared.utils.color_text(
            args.model_from_file, color="blue", method="ansi"
        )
        textattack.shared.logger.info(
            f"Loading model and tokenizer from file: {colored_model_name}"
        )
        if ARGS_SPLIT_TOKEN in args.model_from_file:
            model_file, model_name = args.model_from_file.split(ARGS_SPLIT_TOKEN)
        else:
            _, model_name = args.model_from_file, "model"
        try:
            model_module = load_module_from_file(args.model_from_file)
        except Exception:
            raise ValueError(f"Failed to import file {args.model_from_file}")
        try:
            model = getattr(model_module, model_name)
        except AttributeError:
            raise AttributeError(
                f"``{model_name}`` not found in module {args.model_from_file}"
            )

        if not isinstance(model, textattack.models.wrappers.ModelWrapper):
            raise TypeError(
                "Model must be of type "
                f"``textattack.models.ModelWrapper``, got type {type(model)}"
            )
    elif (args.model in HUGGINGFACE_DATASET_BY_MODEL) or args.model_from_huggingface:
        # Support loading models automatically from the HuggingFace model hub.
        import transformers

        model_name = (
            HUGGINGFACE_DATASET_BY_MODEL[args.model][0]
            if (args.model in HUGGINGFACE_DATASET_BY_MODEL)
            else args.model_from_huggingface
        )

        if ARGS_SPLIT_TOKEN in model_name:
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
        tokenizer = textattack.models.tokenizers.AutoTokenizer(model_name)
        model = textattack.models.wrappers.HuggingFaceModelWrapper(
            model, tokenizer, batch_size=args.model_batch_size
        )
    elif args.model in TEXTATTACK_DATASET_BY_MODEL:
        # Support loading TextAttack pre-trained models via just a keyword.
        model_path, _ = TEXTATTACK_DATASET_BY_MODEL[args.model]
        model = textattack.shared.utils.load_textattack_model_from_path(
            args.model, model_path
        )
        # Choose the approprate model wrapper (based on whether or not this is
        # a HuggingFace model).
        if isinstance(model, textattack.models.helpers.T5ForTextToText):
            model = textattack.models.wrappers.HuggingFaceModelWrapper(
                model, model.tokenizer, batch_size=args.model_batch_size
            )
        else:
            model = textattack.models.wrappers.PyTorchModelWrapper(
                model, model.tokenizer, batch_size=args.model_batch_size
            )
    elif args.model and os.path.exists(args.model):
        # Support loading TextAttack-trained models via just their folder path.
        # If `args.model` is a path/directory, let's assume it was a model
        # trained with textattack, and try and load it.
        model_args_json_path = os.path.join(args.model, "train_args.json")
        if not os.path.exists(model_args_json_path):
            raise FileNotFoundError(
                f"Tried to load model from path {args.model} - could not find train_args.json."
            )
        model_train_args = json.loads(open(model_args_json_path).read())
        if model_train_args["model"] not in {"cnn", "lstm"}:
            # for huggingface models, set args.model to the path of the model
            model_train_args["model"] = args.model
        num_labels = model_train_args["num_labels"]
        from textattack.commands.train_model.train_args_helpers import model_from_args

        model = model_from_args(
            argparse.Namespace(**model_train_args),
            num_labels,
            model_path=args.model,
        )

    else:
        raise ValueError(f"Error: unsupported TextAttack model {args.model}")

    return model


def parse_dataset_from_args(args):
    # Automatically detect dataset for huggingface & textattack models.
    # This allows us to use the --model shortcut without specifying a dataset.
    if args.model in HUGGINGFACE_DATASET_BY_MODEL:
        _, args.dataset_from_huggingface = HUGGINGFACE_DATASET_BY_MODEL[args.model]
    elif args.model in TEXTATTACK_DATASET_BY_MODEL:
        _, dataset = TEXTATTACK_DATASET_BY_MODEL[args.model]
        if dataset[0].startswith("textattack"):
            # unsavory way to pass custom dataset classes
            # ex: dataset = ('textattack.datasets.translation.TedMultiTranslationDataset', 'en', 'de')
            dataset = eval(f"{dataset[0]}")(*dataset[1:])
            return dataset
        else:
            args.dataset_from_huggingface = dataset
    # Automatically detect dataset for models trained with textattack.
    elif args.model and os.path.exists(args.model):
        model_args_json_path = os.path.join(args.model, "train_args.json")
        if not os.path.exists(model_args_json_path):
            raise FileNotFoundError(
                f"Tried to load model from path {args.model} - could not find train_args.json."
            )
        model_train_args = json.loads(open(model_args_json_path).read())
        try:
            if ARGS_SPLIT_TOKEN in model_train_args["dataset"]:
                name, subset = model_train_args["dataset"].split(ARGS_SPLIT_TOKEN)
            else:
                name, subset = model_train_args["dataset"], None
            args.dataset_from_huggingface = (
                name,
                subset,
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
        if ARGS_SPLIT_TOKEN in args.dataset_from_file:
            dataset_file, dataset_name = args.dataset_from_file.split(ARGS_SPLIT_TOKEN)
        else:
            dataset_file, dataset_name = args.dataset_from_file, "dataset"
        try:
            dataset_module = load_module_from_file(dataset_file)
        except Exception:
            raise ValueError(
                f"Failed to import dataset from file {args.dataset_from_file}"
            )
        try:
            dataset = getattr(dataset_module, dataset_name)
        except AttributeError:
            raise AttributeError(
                f"``dataset`` not found in module {args.dataset_from_file}"
            )
    elif args.dataset_from_huggingface:
        dataset_args = args.dataset_from_huggingface
        if isinstance(dataset_args, str):
            if ARGS_SPLIT_TOKEN in dataset_args:
                dataset_args = dataset_args.split(ARGS_SPLIT_TOKEN)
            else:
                dataset_args = (dataset_args,)
        dataset = textattack.datasets.HuggingFaceDataset(
            *dataset_args, shuffle=args.shuffle
        )
        dataset.examples = dataset.examples[args.num_examples_offset :]
    else:
        raise ValueError("Must supply pretrained model or dataset")
    if args.num_examples == -1 or args.num_examples > len(dataset):
        args.num_examples = len(dataset)
    return dataset


def parse_logger_from_args(args):
    # Create logger
    attack_log_manager = textattack.loggers.AttackLogManager()

    # Get current time for file naming
    timestamp = time.strftime("%Y-%m-%d-%H-%M")

    # Get default directory to save results
    current_dir = os.path.dirname(os.path.realpath(__file__))
    outputs_dir = os.path.join(
        current_dir, os.pardir, os.pardir, os.pardir, "outputs", "attacks"
    )
    out_dir_txt = out_dir_csv = os.path.normpath(outputs_dir)

    # Get default txt and csv file names
    if args.recipe:
        filename_txt = f"{args.model}_{args.recipe}_{timestamp}.txt"
        filename_csv = f"{args.model}_{args.recipe}_{timestamp}.csv"
    else:
        filename_txt = f"{args.model}_{timestamp}.txt"
        filename_csv = f"{args.model}_{timestamp}.csv"

    # if '--log-to-txt' specified with arguments
    if args.log_to_txt:
        # if user decide to save to a specific directory
        if args.log_to_txt[-1] == "/":
            out_dir_txt = args.log_to_txt
        # else if path + filename is given
        elif args.log_to_txt[-4:] == ".txt":
            out_dir_txt = args.log_to_txt.rsplit("/", 1)[0]
            filename_txt = args.log_to_txt.rsplit("/", 1)[-1]
        # otherwise, customize filename
        else:
            filename_txt = f"{args.log_to_txt}.txt"

    # if "--log-to-csv" is called
    if args.log_to_csv:
        # if user decide to save to a specific directory
        if args.log_to_csv[-1] == "/":
            out_dir_csv = args.log_to_csv
        # else if path + filename is given
        elif args.log_to_csv[-4:] == ".csv":
            out_dir_csv = args.log_to_csv.rsplit("/", 1)[0]
            filename_csv = args.log_to_csv.rsplit("/", 1)[-1]
        # otherwise, customize filename
        else:
            filename_csv = f"{args.log_to_csv}.csv"

    # in case directory doesn't exist
    if not os.path.exists(out_dir_txt):
        os.makedirs(out_dir_txt)
    if not os.path.exists(out_dir_csv):
        os.makedirs(out_dir_csv)

    # if "--log-to-txt" specified in terminal command (with or without arg), save to a txt file
    if args.log_to_txt == "" or args.log_to_txt:
        attack_log_manager.add_output_file(os.path.join(out_dir_txt, filename_txt))

    # if "--log-to-csv" specified in terminal command(with	or without arg), save to a csv file
    if args.log_to_csv == "" or args.log_to_csv:
        # "--csv-style used to swtich from 'fancy' to 'plain'
        color_method = None if args.csv_style == "plain" else "file"
        csv_path = os.path.join(out_dir_csv, filename_csv)
        attack_log_manager.add_output_csv(csv_path, color_method)
        textattack.shared.logger.info(f"Logging to CSV at path {csv_path}.")

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
    """Merge previously saved arguments for checkpoint and newly entered
    arguments."""
    args = copy.deepcopy(saved_args)
    # Newly entered arguments take precedence
    args.parallel = cmdline_args.parallel
    # If set, replace
    if cmdline_args.checkpoint_dir:
        args.checkpoint_dir = cmdline_args.checkpoint_dir
    if cmdline_args.checkpoint_interval:
        args.checkpoint_interval = cmdline_args.checkpoint_interval

    return args
