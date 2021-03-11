from dataclasses import dataclass, field
import json
import os
import time

import textattack
from textattack.shared.utils import ARGS_SPLIT_TOKEN, load_module_from_file

from .attack import Attack
from .dataset_args import DatasetArgs
from .model_args import ModelArgs

ATTACK_RECIPE_NAMES = {
    "alzantot": "textattack.attack_recipes.GeneticAlgorithmAlzantot2018",
    "bae": "textattack.attack_recipes.BAEGarg2019",
    "bert-attack": "textattack.attack_recipes.BERTAttackLi2020",
    "faster-alzantot": "textattack.attack_recipes.FasterGeneticAlgorithmJia2019",
    "deepwordbug": "textattack.attack_recipes.DeepWordBugGao2018",
    "hotflip": "textattack.attack_recipes.HotFlipEbrahimi2017",
    "input-reduction": "textattack.attack_recipes.InputReductionFeng2018",
    "kuleshov": "textattack.attack_recipes.Kuleshov2017",
    "morpheus": "textattack.attack_recipes.MorpheusTan2020",
    "seq2sick": "textattack.attack_recipes.Seq2SickCheng2018BlackBox",
    "textbugger": "textattack.attack_recipes.TextBuggerLi2018",
    "textfooler": "textattack.attack_recipes.TextFoolerJin2019",
    "pwws": "textattack.attack_recipes.PWWSRen2019",
    "iga": "textattack.attack_recipes.IGAWang2019",
    "pruthi": "textattack.attack_recipes.Pruthi2019",
    "pso": "textattack.attack_recipes.PSOZang2020",
    "checklist": "textattack.attack_recipes.CheckList2020",
    "clare": "textattack.attack_recipes.CLARE2020",
}


BLACK_BOX_TRANSFORMATION_CLASS_NAMES = {
    "random-synonym-insertion": "textattack.transformations.RandomSynonymInsertion",
    "word-deletion": "textattack.transformations.WordDeletion",
    "word-swap-embedding": "textattack.transformations.WordSwapEmbedding",
    "word-swap-homoglyph": "textattack.transformations.WordSwapHomoglyphSwap",
    "word-swap-inflections": "textattack.transformations.WordSwapInflections",
    "word-swap-neighboring-char-swap": "textattack.transformations.WordSwapNeighboringCharacterSwap",
    "word-swap-random-char-deletion": "textattack.transformations.WordSwapRandomCharacterDeletion",
    "word-swap-random-char-insertion": "textattack.transformations.WordSwapRandomCharacterInsertion",
    "word-swap-random-char-substitution": "textattack.transformations.WordSwapRandomCharacterSubstitution",
    "word-swap-wordnet": "textattack.transformations.WordSwapWordNet",
    "word-swap-masked-lm": "textattack.transformations.WordSwapMaskedLM",
    "word-swap-hownet": "textattack.transformations.WordSwapHowNet",
    "word-swap-qwerty": "textattack.transformations.WordSwapQWERTY",
}


WHITE_BOX_TRANSFORMATION_CLASS_NAMES = {
    "word-swap-gradient": "textattack.transformations.WordSwapGradientBased"
}


CONSTRAINT_CLASS_NAMES = {
    #
    # Semantics constraints
    #
    "embedding": "textattack.constraints.semantics.WordEmbeddingDistance",
    "bert": "textattack.constraints.semantics.sentence_encoders.BERT",
    "infer-sent": "textattack.constraints.semantics.sentence_encoders.InferSent",
    "thought-vector": "textattack.constraints.semantics.sentence_encoders.ThoughtVector",
    "use": "textattack.constraints.semantics.sentence_encoders.UniversalSentenceEncoder",
    "muse": "textattack.constraints.semantics.sentence_encoders.MultilingualUniversalSentenceEncoder",
    "bert-score": "textattack.constraints.semantics.BERTScore",
    #
    # Grammaticality constraints
    #
    "lang-tool": "textattack.constraints.grammaticality.LanguageTool",
    "part-of-speech": "textattack.constraints.grammaticality.PartOfSpeech",
    "goog-lm": "textattack.constraints.grammaticality.language_models.GoogleLanguageModel",
    "gpt2": "textattack.constraints.grammaticality.language_models.GPT2",
    "learning-to-write": "textattack.constraints.grammaticality.language_models.LearningToWriteLanguageModel",
    "cola": "textattack.constraints.grammaticality.COLA",
    #
    # Overlap constraints
    #
    "bleu": "textattack.constraints.overlap.BLEU",
    "chrf": "textattack.constraints.overlap.chrF",
    "edit-distance": "textattack.constraints.overlap.LevenshteinEditDistance",
    "meteor": "textattack.constraints.overlap.METEOR",
    "max-words-perturbed": "textattack.constraints.overlap.MaxWordsPerturbed",
    #
    # Pre-transformation constraints
    #
    "repeat": "textattack.constraints.pre_transformation.RepeatModification",
    "stopword": "textattack.constraints.pre_transformation.StopwordModification",
    "max-word-index": "textattack.constraints.pre_transformation.MaxWordIndexModification",
}


SEARCH_METHOD_CLASS_NAMES = {
    "beam-search": "textattack.search_methods.BeamSearch",
    "greedy": "textattack.search_methods.GreedySearch",
    "ga-word": "textattack.search_methods.GeneticAlgorithm",
    "greedy-word-wir": "textattack.search_methods.GreedyWordSwapWIR",
    "pso": "textattack.search_methods.ParticleSwarmOptimization",
}


GOAL_FUNCTION_CLASS_NAMES = {
    "input-reduction": "textattack.goal_functions.InputReduction",
    "minimize-bleu": "textattack.goal_functions.MinimizeBleu",
    "non-overlapping-output": "textattack.goal_functions.NonOverlappingOutput",
    "targeted-classification": "textattack.goal_functions.TargetedClassification",
    "untargeted-classification": "textattack.goal_functions.UntargetedClassification",
}


@dataclass
class AttackArgs:
    """Attack args for running attacks via API. This assumes that ``Attack``
    has already been created by the user.

    Args:
        num_examples (int): The number of examples to attack. -1 for entire dataset.
        num_examples_offset (int): The offset to start at in the dataset.
        query_budget (int): The maximum number of model queries allowed per example attacked.
            This is optional and setting this overwrites the query budget set in `GoalFunction` object.
        shuffle (bool): If `True`, shuffle the samples before we attack the dataset. Note this does not involve shuffling the dataset internally. Default is False.
        attack_n (bool): Whether to run attack until total of `n` examples have been attacked (not skipped).
        checkpoint_dir (str): The directory to save checkpoint files.
        checkpoint_interval (int): If set, checkpoint will be saved after attacking every N examples. If not set, no checkpoints will be saved.
        random_seed (int): Random seed for reproducibility. Default is 765.
        parallel (bool): Run attack using multiple CPUs/GPUs.
        num_workers_per_device (int): Number of worker processes to run per device. For example, if you are using GPUs and ``num_workers_per_device=2``,
            then 2 processes will be running in each GPU. If you are only using CPU, then this is equivalent to running 2 processes concurrently.
        log_to_txt (str): Path to which to save attack logs as a text file. Set this argument if you want to save text logs.
            If the last part of the path ends with `.txt` extension, the path is assumed to path for output file.
        log_to_csv (str): Path to which to save attack logs as a CSV file. Set this argument if you want to save CSV logs.
            If the last part of the path ends with `.csv` extension, the path is assumed to path for output file.
        csv_coloring_style (str): Method for choosing how to mark perturbed parts of the text. Options are "file" and "plain".
            "file" wraps text with double brackets `[[ <text> ]]` while "plain" does not mark any text. Default is "file".
        log_to_visdom (dict): Set this argument if you want to log attacks to Visdom. The dictionary should have the following
            three keys and their corresponding values: `"env", "port", "hostname"` (e.g. `{"env": "main", "port": 8097, "hostname": "localhost"}`).
        log_to_wandb (str): Name of the wandb project. Set this argument if you want to log attacks to Wandb.
        disable_stdout (bool): Disable logging attack results to stdout.
        silent (bool): Disable all logging.
        ignore_exceptions (bool): Skip examples that raise an error instead of exiting.
    """

    num_examples: int = 5
    num_examples_offset: int = 0
    query_budget: int = None
    shuffle: bool = False
    attack_n: bool = False
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = None
    random_seed: int = 765  # equivalent to sum((ord(c) for c in "TEXTATTACK"))
    parallel: bool = False
    num_workers_per_device: int = 1
    log_to_txt: str = None
    log_to_csv: str = None
    csv_coloring_style: str = "file"
    log_to_visdom: dict = None
    log_to_wandb: str = None
    disable_stdout: bool = False
    silent: bool = False
    ignore_exceptions: bool = False

    @classmethod
    def add_parser_args(cls, parser):
        """Add listed args to command line parser."""
        parser.add_argument(
            "--num-examples",
            "-n",
            type=int,
            required=False,
            default=5,
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
        parser.add_argument(
            "--query-budget",
            "-q",
            type=int,
            default=None,
            help="The maximum number of model queries allowed per example attacked. Setting this overwrites the query budget set in `GoalFunction` object.",
        )
        parser.add_argument(
            "--shuffle",
            action="store_true",
            default=False,
            help="If `True`, shuffle the samples before we attack the dataset. Default is False.",
        )
        parser.add_argument(
            "--attack-n",
            action="store_true",
            default=False,
            help="Whether to run attack until `n` examples have been attacked (not skipped).",
        )
        parser.add_argument(
            "--checkpoint-dir",
            required=False,
            type=str,
            default="checkpoints",
            help="The directory to save checkpoint files.",
        )
        parser.add_argument(
            "--checkpoint-interval",
            required=False,
            type=int,
            help="If set, checkpoint will be saved after attacking every N examples. If not set, no checkpoints will be saved.",
        )

        def str_to_int(s):
            return sum((ord(c) for c in s))

        parser.add_argument("--random-seed", default=str_to_int("TEXTATTACK"), type=int)

        parser.add_argument(
            "--parallel",
            action="store_true",
            default=False,
            help="Run attack using multiple GPUs.",
        )
        parser.add_argument(
            "--num-workers-per-device",
            default=1,
            type=int,
            help="Number of worker processes to run per device.",
        )

        parser.add_argument(
            "--log-to-txt",
            nargs="?",
            default=None,
            const="",
            type=str,
            help="Path to which to save attack logs as a text file. Set this argument if you want to save text logs. "
            "If the last part of the path ends with `.txt` extension, the path is assumed to path for output file.",
        )

        parser.add_argument(
            "--log-to-csv",
            nargs="?",
            default=None,
            const="",
            type=str,
            help="Path to which to save attack logs as a CSV file. Set this argument if you want to save CSV logs. "
            "If the last part of the path ends with `.csv` extension, the path is assumed to path for output file.",
        )

        parser.add_argument(
            "--csv-coloring-style",
            default="file",
            type=str,
            help='Method for choosing how to mark perturbed parts of the text in CSV logs. Options are "file" and "plain". '
            '"file" wraps text with double brackets `[[ <text> ]]` while "plain" does not mark any text. Default is "file".',
        )

        parser.add_argument(
            "--log-to-visdom",
            nargs="?",
            default=None,
            const='{"env": "main", "port": 8097, "hostname": "localhost"}',
            type=json.loads,
            help="Set this argument if you want to log attacks to Visdom. The dictionary should have the following "
            'three keys and their corresponding values: `"env", "port", "hostname"`. '
            'Example for command line use: `--log-to-visdom {"env": "main", "port": 8097, "hostname": "localhost"}`.',
        )

        parser.add_argument(
            "--log-to-wandb",
            nargs="?",
            default=None,
            const="textattack",
            type=str,
            help="Name of the wandb project. Set this argument if you want to log attacks to Wandb.",
        )

        parser.add_argument(
            "--disable-stdout",
            action="store_true",
            help="Disable logging attack results to stdout",
        )

        parser.add_argument(
            "--silent", action="store_true", default=False, help="Disable all logging"
        )
        parser.add_argument(
            "--ignore-exceptions",
            action="store_true",
            default=False,
            help="Skip examples that raise an error instead of exiting.",
        )

        return parser

    @classmethod
    def create_loggers_from_args(cls, args):
        assert isinstance(
            args, cls
        ), f"Expect args to be of type `{type(cls)}`, but got type `{type(args)}`."

        # Create logger
        attack_log_manager = textattack.loggers.AttackLogManager()

        # Get current time for file naming
        timestamp = time.strftime("%Y-%m-%d-%H-%M")

        # if '--log-to-txt' specified with arguments
        if args.log_to_txt is not None:
            if args.log_to_txt.lower().endswith(".txt"):
                txt_file_path = args.log_to_txt
            else:
                txt_file_path = os.path.join(args.log_to_txt, f"{timestamp}-log.txt")

            dir_path = os.path.dirname(txt_file_path)
            dir_path = dir_path if dir_path else "."
            if not os.path.exists(dir_path):
                os.makedirs(os.path.dirname(txt_file_path))

            attack_log_manager.add_output_file(txt_file_path)

        # if '--log-to-csv' specified with arguments
        if args.log_to_csv is not None:
            if args.log_to_csv.lower().endswith(".csv"):
                csv_file_path = args.log_to_csv
            else:
                csv_file_path = os.path.join(args.log_to_csv, f"{timestamp}-log.csv")

            dir_path = os.path.dirname(csv_file_path)
            dir_path = dir_path if dir_path else "."
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            color_method = (
                None if args.csv_coloring_style == "plain" else args.csv_coloring_style
            )
            attack_log_manager.add_output_csv(csv_file_path, color_method)

        # Visdom
        if args.log_to_visdom is not None:
            attack_log_manager.enable_visdom(**args.log_to_visdom)

        # Weights & Biases
        if args.log_to_wandb is not None:
            attack_log_manager.enable_wandb(args.log_to_wandb)

        # Stdout
        if not args.disable_stdout:
            attack_log_manager.enable_stdout()

        return attack_log_manager


@dataclass
class _CommandLineAttackArgs:
    """Command line interface attack args. This requires more arguments to
    create ``Attack`` object as specified.

    Args:
        transformation (str): Name of transformation to use.
        constraints (list[str]): List of names of constraints to use.
        goal_function (str): Name of goal function to use.
        search_method (str): Name of search method to use.
        attack_recipe (str): Name of attack recipe to use.
            If this is set, it overrides any previous selection of transformation, constraints, goal function, and search method.
        attack_from_file (str): Path of `.py` file from which to load attack from. Use `<path>^<variable_name>` to specifiy which variable to import from the file.
            If this is set, it overrides any previous selection of transformation, constraints, goal function, and search method
        interactive (bool): If `True`, carry attack in interactive mode. Default is `False`.
        parallel (bool): If `True`, attack in parallel. Default is `False`.
        model_batch_size (int): The batch size for making calls to the model.
        model_cache_size (int): The maximum number of items to keep in the model results cache at once.
        constraint-cache-size (int): The maximum number of items to keep in the constraints cache at once.
    """

    transformation: str = "word-swap-embedding"
    constraints: list = field(default_factory=lambda: ["repeat", "stopword"])
    goal_function: str = "untargeted-classification"
    search_method: str = "greedy-word-wir"
    attack_recipe: str = None
    attack_from_file: str = None
    interactive: bool = False
    parallel: bool = False
    model_batch_size: int = 32
    model_cache_size: int = 2 ** 18
    constraint_cache_size: int = 2 ** 18

    @classmethod
    def add_parser_args(cls, parser):
        """Add listed args to command line parser."""
        transformation_names = set(BLACK_BOX_TRANSFORMATION_CLASS_NAMES.keys()) | set(
            WHITE_BOX_TRANSFORMATION_CLASS_NAMES.keys()
        )
        parser.add_argument(
            "--transformation",
            type=str,
            required=False,
            default="word-swap-embedding",
            help='The transformation to apply. Usage: "--transformation {transformation}:{arg_1}={value_1},{arg_3}={value_3}". Choices: '
            + str(transformation_names),
        )
        parser.add_argument(
            "--constraints",
            type=str,
            required=False,
            nargs="*",
            default=["repeat", "stopword"],
            help='Constraints to add to the attack. Usage: "--constraints {constraint}:{arg_1}={value_1},{arg_3}={value_3}". Choices: '
            + str(CONSTRAINT_CLASS_NAMES.keys()),
        )
        goal_function_choices = ", ".join(GOAL_FUNCTION_CLASS_NAMES.keys())
        parser.add_argument(
            "--goal-function",
            "-g",
            default="untargeted-classification",
            help=f"The goal function to use. choices: {goal_function_choices}",
        )
        attack_group = parser.add_mutually_exclusive_group(required=False)
        search_choices = ", ".join(SEARCH_METHOD_CLASS_NAMES.keys())
        attack_group.add_argument(
            "--search-method",
            "--search",
            "-s",
            type=str,
            required=False,
            default="greedy-word-wir",
            help=f"The search method to use. choices: {search_choices}",
        )
        attack_group.add_argument(
            "--attack-recipe",
            "--recipe",
            "-r",
            type=str,
            required=False,
            default=None,
            help="full attack recipe (overrides provided goal function, transformation & constraints)",
            choices=ATTACK_RECIPE_NAMES.keys(),
        )
        attack_group.add_argument(
            "--attack-from-file",
            type=str,
            required=False,
            default=None,
            help="Path of `.py` file from which to load attack from. Use `<path>^<variable_name>` to specifiy which variable to import from the file.",
        )
        parser.add_argument(
            "--interactive",
            action="store_true",
            default=False,
            help="Whether to run attacks interactively.",
        )
        parser.add_argument(
            "--model-batch-size",
            type=int,
            default=32,
            help="The batch size for making calls to the model.",
        )
        parser.add_argument(
            "--model-cache-size",
            type=int,
            default=2 ** 18,
            help="The maximum number of items to keep in the model results cache at once.",
        )
        parser.add_argument(
            "--constraint-cache-size",
            type=int,
            default=2 ** 18,
            help="The maximum number of items to keep in the constraints cache at once.",
        )

        return parser

    @classmethod
    def _create_transformation_from_args(cls, args, model_wrapper):
        """Create `Transformation` based on provided `args` and
        `model_wrapper`."""

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
                raise ValueError(
                    f"Error: unsupported transformation {transformation_name}"
                )
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
                raise ValueError(
                    f"Error: unsupported transformation {transformation_name}"
                )
        return transformation

    @classmethod
    def _create_goal_function_from_args(cls, args, model_wrapper):
        """Create `GoalFunction` based on provided `args` and
        `model_wrapper`."""

        goal_function = args.goal_function
        if ARGS_SPLIT_TOKEN in goal_function:
            goal_function_name, params = goal_function.split(ARGS_SPLIT_TOKEN)
            if goal_function_name not in GOAL_FUNCTION_CLASS_NAMES:
                raise ValueError(
                    f"Error: unsupported goal_function {goal_function_name}"
                )
            goal_function = eval(
                f"{GOAL_FUNCTION_CLASS_NAMES[goal_function_name]}(model_wrapper, {params})"
            )
        elif goal_function in GOAL_FUNCTION_CLASS_NAMES:
            goal_function = eval(
                f"{GOAL_FUNCTION_CLASS_NAMES[goal_function]}(model_wrapper)"
            )
        else:
            raise ValueError(f"Error: unsupported goal_function {goal_function}")
        if args.query_budget:
            goal_function.query_budget = args.query_budget
        goal_function.model_cache_size = args.model_cache_size
        goal_function.batch_size = args.model_batch_size
        return goal_function

    @classmethod
    def _create_constraints_from_args(cls, args):
        """Create list of `Constraints` based on provided `args`."""

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

    @classmethod
    def create_attack_from_args(cls, args, model_wrapper):
        """Given ``CommandLineArgs`` and ``ModelWrapper``, return specified
        ``Attack`` object."""

        assert isinstance(
            args, cls
        ), f"Expect args to be of type `{type(cls)}`, but got type `{type(args)}`."

        if args.attack_recipe:
            if ARGS_SPLIT_TOKEN in args.attack_recipe:
                recipe_name, params = args.attack_recipe.split(ARGS_SPLIT_TOKEN)
                if recipe_name not in ATTACK_RECIPE_NAMES:
                    raise ValueError(f"Error: unsupported recipe {recipe_name}")
                recipe = eval(
                    f"{ATTACK_RECIPE_NAMES[recipe_name]}.build(model_wrapper, {params})"
                )
            elif args.attack_recipe in ATTACK_RECIPE_NAMES:
                recipe = eval(
                    f"{ATTACK_RECIPE_NAMES[args.attack_recipe]}.build(model_wrapper)"
                )
            else:
                raise ValueError(f"Invalid recipe {args.attack_recipe}")
            if args.query_budget:
                recipe.goal_function.query_budget = args.query_budget
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
            return attack_func(model_wrapper)
        else:
            goal_function = cls._create_goal_function_from_args(args, model_wrapper)
            transformation = cls._create_transformation_from_args(args, model_wrapper)
            constraints = cls._create_constraints_from_args(args)
            if ARGS_SPLIT_TOKEN in args.search_method:
                search_name, params = args.search_method.split(ARGS_SPLIT_TOKEN)
                if search_name not in SEARCH_METHOD_CLASS_NAMES:
                    raise ValueError(f"Error: unsupported search {search_name}")
                search_method = eval(
                    f"{SEARCH_METHOD_CLASS_NAMES[search_name]}({params})"
                )
            elif args.search_method in SEARCH_METHOD_CLASS_NAMES:
                search_method = eval(
                    f"{SEARCH_METHOD_CLASS_NAMES[args.search_method]}()"
                )
            else:
                raise ValueError(f"Error: unsupported attack {args.search_method}")

        return Attack(
            goal_function,
            constraints,
            transformation,
            search_method,
            constraint_cache_size=args.constraint_cache_size,
        )


# This neat trick allows use to reorder the arguments to avoid TypeErrors commonly found when inheriting dataclass.
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
@dataclass
class CommandLineAttackArgs(AttackArgs, _CommandLineAttackArgs, DatasetArgs, ModelArgs):
    @classmethod
    def add_parser_args(cls, parser):
        """Add listed args to command line parser."""
        parser = ModelArgs.add_parser_args(parser)
        parser = DatasetArgs.add_parser_args(parser)
        parser = _CommandLineAttackArgs.add_parser_args(parser)
        parser = AttackArgs.add_parser_args(parser)
        return parser
