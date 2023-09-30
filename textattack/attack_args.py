"""
AttackArgs Class
================
"""

from dataclasses import dataclass, field
import json
import os
import sys
import time
from typing import Dict, Optional

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
    "a2t": "textattack.attack_recipes.A2TYoo2021",
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
    #
    # Classification goal functions
    #
    "hardlabel-classification": "textattack.goal_functions.classification.HardLabelClassification",
    "targeted-classification": "textattack.goal_functions.classification.TargetedClassification",
    "untargeted-classification": "textattack.goal_functions.classification.UntargetedClassification",
    "input-reduction": "textattack.goal_functions.classification.InputReduction",
    #
    # Text goal functions
    #
    "minimize-bleu": "textattack.goal_functions.text.MinimizeBleu",
    "non-overlapping-output": "textattack.goal_functions.text.NonOverlappingOutput",
    "text-to-text": "textattack.goal_functions.text.TextToTextGoalFunction",
}


@dataclass
class AttackArgs:
    """Attack arguments to be passed to :class:`~textattack.Attacker`.

    Args:
        num_examples (:obj:`int`, 'optional`, defaults to :obj:`10`):
            The number of examples to attack. :obj:`-1` for entire dataset.
        num_successful_examples (:obj:`int`, `optional`, defaults to :obj:`None`):
            The number of successful adversarial examples we want. This is different from :obj:`num_examples`
            as :obj:`num_examples` only cares about attacking `N` samples while :obj:`num_successful_examples` aims to keep attacking
            until we have `N` successful cases.
            .. note::
                If set, this argument overrides `num_examples` argument.
        num_examples_offset (:obj: `int`, `optional`, defaults to :obj:`0`):
            The offset index to start at in the dataset.
        attack_n (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run attack until total of `N` examples have been attacked (and not skipped).
        shuffle (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, we randomly shuffle the dataset before attacking. However, this avoids actually shuffling
            the dataset internally and opts for shuffling the list of indices of examples we want to attack. This means
            :obj:`shuffle` can now be used with checkpoint saving.
        query_budget (:obj:`int`, `optional`, defaults to :obj:`None`):
            The maximum number of model queries allowed per example attacked.
            If not set, we use the query budget set in the :class:`~textattack.goal_functions.GoalFunction` object (which by default is :obj:`float("inf")`).
            .. note::
                Setting this overwrites the query budget set in :class:`~textattack.goal_functions.GoalFunction` object.
        checkpoint_interval (:obj:`int`, `optional`, defaults to :obj:`None`):
            If set, checkpoint will be saved after attacking every `N` examples. If :obj:`None` is passed, no checkpoints will be saved.
        checkpoint_dir (:obj:`str`, `optional`, defaults to :obj:`"checkpoints"`):
            The directory to save checkpoint files.
        random_seed (:obj:`int`, `optional`, defaults to :obj:`765`):
            Random seed for reproducibility.
        parallel (:obj:`False`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, run attack using multiple CPUs/GPUs.
        num_workers_per_device (:obj:`int`, `optional`, defaults to :obj:`1`):
            Number of worker processes to run per device in parallel mode (i.e. :obj:`parallel=True`). For example, if you are using GPUs and :obj:`num_workers_per_device=2`,
            then 2 processes will be running in each GPU.
        log_to_txt (:obj:`str`, `optional`, defaults to :obj:`None`):
            If set, save attack logs as a `.txt` file to the directory specified by this argument.
            If the last part of the provided path ends with `.txt` extension, it is assumed to the desired path of the log file.
        log_to_csv (:obj:`str`, `optional`, defaults to :obj:`None`):
            If set, save attack logs as a CSV file to the directory specified by this argument.
            If the last part of the provided path ends with `.csv` extension, it is assumed to the desired path of the log file.
        csv_coloring_style (:obj:`str`, `optional`, defaults to :obj:`"file"`):
            Method for choosing how to mark perturbed parts of the text. Options are :obj:`"file"`, :obj:`"plain"`, and :obj:`"html"`.
            :obj:`"file"` wraps perturbed parts with double brackets :obj:`[[ <text> ]]` while :obj:`"plain"` does not mark the text in any way.
        log_to_visdom (:obj:`dict`, `optional`, defaults to :obj:`None`):
            If set, Visdom logger is used with the provided dictionary passed as a keyword arguments to :class:`~textattack.loggers.VisdomLogger`.
            Pass in empty dictionary to use default arguments. For custom logger, the dictionary should have the following
            three keys and their corresponding values: :obj:`"env", "port", "hostname"`.
        log_to_wandb(:obj:`dict`, `optional`, defaults to :obj:`None`):
            If set, WandB logger is used with the provided dictionary passed as a keyword arguments to :class:`~textattack.loggers.WeightsAndBiasesLogger`.
            Pass in empty dictionary to use default arguments. For custom logger, the dictionary should have the following
            key and its corresponding value: :obj:`"project"`.
        disable_stdout (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Disable displaying individual attack results to stdout.
        silent (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Disable all logging (except for errors). This is stronger than :obj:`disable_stdout`.
        enable_advance_metrics (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Enable calculation and display of optional advance post-hoc metrics like perplexity, grammar errors, etc.
    """

    num_examples: int = 10
    num_successful_examples: int = None
    num_examples_offset: int = 0
    attack_n: bool = False
    shuffle: bool = False
    query_budget: int = None
    checkpoint_interval: int = None
    checkpoint_dir: str = "checkpoints"
    random_seed: int = 765  # equivalent to sum((ord(c) for c in "TEXTATTACK"))
    parallel: bool = False
    num_workers_per_device: int = 1
    log_to_txt: str = None
    log_to_csv: str = None
    log_summary_to_json: str = None
    csv_coloring_style: str = "file"
    log_to_visdom: dict = None
    log_to_wandb: dict = None
    disable_stdout: bool = False
    silent: bool = False
    enable_advance_metrics: bool = False
    metrics: Optional[Dict] = None

    def __post_init__(self):
        if self.num_successful_examples:
            self.num_examples = None
        if self.num_examples:
            assert (
                self.num_examples >= 0 or self.num_examples == -1
            ), "`num_examples` must be greater than or equal to 0 or equal to -1."
        if self.num_successful_examples:
            assert (
                self.num_successful_examples >= 0
            ), "`num_examples` must be greater than or equal to 0."

        if self.query_budget:
            assert self.query_budget > 0, "`query_budget` must be greater than 0."

        if self.checkpoint_interval:
            assert (
                self.checkpoint_interval > 0
            ), "`checkpoint_interval` must be greater than 0."

        assert (
            self.num_workers_per_device > 0
        ), "`num_workers_per_device` must be greater than 0."

    @classmethod
    def _add_parser_args(cls, parser):
        """Add listed args to command line parser."""
        default_obj = cls()
        num_ex_group = parser.add_mutually_exclusive_group(required=False)
        num_ex_group.add_argument(
            "--num-examples",
            "-n",
            type=int,
            default=default_obj.num_examples,
            help="The number of examples to process, -1 for entire dataset.",
        )
        num_ex_group.add_argument(
            "--num-successful-examples",
            type=int,
            default=default_obj.num_successful_examples,
            help="The number of successful adversarial examples we want.",
        )
        parser.add_argument(
            "--num-examples-offset",
            "-o",
            type=int,
            required=False,
            default=default_obj.num_examples_offset,
            help="The offset to start at in the dataset.",
        )
        parser.add_argument(
            "--query-budget",
            "-q",
            type=int,
            default=default_obj.query_budget,
            help="The maximum number of model queries allowed per example attacked. Setting this overwrites the query budget set in `GoalFunction` object.",
        )
        parser.add_argument(
            "--shuffle",
            action="store_true",
            default=default_obj.shuffle,
            help="If `True`, shuffle the samples before we attack the dataset. Default is False.",
        )
        parser.add_argument(
            "--attack-n",
            action="store_true",
            default=default_obj.attack_n,
            help="Whether to run attack until `n` examples have been attacked (not skipped).",
        )
        parser.add_argument(
            "--checkpoint-dir",
            required=False,
            type=str,
            default=default_obj.checkpoint_dir,
            help="The directory to save checkpoint files.",
        )
        parser.add_argument(
            "--checkpoint-interval",
            required=False,
            type=int,
            default=default_obj.checkpoint_interval,
            help="If set, checkpoint will be saved after attacking every N examples. If not set, no checkpoints will be saved.",
        )
        parser.add_argument(
            "--random-seed",
            default=default_obj.random_seed,
            type=int,
            help="Random seed for reproducibility.",
        )
        parser.add_argument(
            "--parallel",
            action="store_true",
            default=default_obj.parallel,
            help="Run attack using multiple GPUs.",
        )
        parser.add_argument(
            "--num-workers-per-device",
            default=default_obj.num_workers_per_device,
            type=int,
            help="Number of worker processes to run per device.",
        )
        parser.add_argument(
            "--log-to-txt",
            nargs="?",
            default=default_obj.log_to_txt,
            const="",
            type=str,
            help="Path to which to save attack logs as a text file. Set this argument if you want to save text logs. "
            "If the last part of the path ends with `.txt` extension, the path is assumed to path for output file.",
        )
        parser.add_argument(
            "--log-to-csv",
            nargs="?",
            default=default_obj.log_to_csv,
            const="",
            type=str,
            help="Path to which to save attack logs as a CSV file. Set this argument if you want to save CSV logs. "
            "If the last part of the path ends with `.csv` extension, the path is assumed to path for output file.",
        )
        parser.add_argument(
            "--log-summary-to-json",
            nargs="?",
            default=default_obj.log_summary_to_json,
            const="",
            type=str,
            help="Path to which to save attack summary as a JSON file. Set this argument if you want to save attack results summary in a JSON. "
            "If the last part of the path ends with `.json` extension, the path is assumed to path for output file.",
        )
        parser.add_argument(
            "--csv-coloring-style",
            default=default_obj.csv_coloring_style,
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
            const='{"project": "textattack"}',
            type=json.loads,
            help="Set this argument if you want to log attacks to WandB. The dictionary should have the following "
            'key and its corresponding value: `"project"`. '
            'Example for command line use: `--log-to-wandb {"project": "textattack"}`.',
        )
        parser.add_argument(
            "--disable-stdout",
            action="store_true",
            default=default_obj.disable_stdout,
            help="Disable logging attack results to stdout",
        )
        parser.add_argument(
            "--silent",
            action="store_true",
            default=default_obj.silent,
            help="Disable all logging",
        )
        parser.add_argument(
            "--enable-advance-metrics",
            action="store_true",
            default=default_obj.enable_advance_metrics,
            help="Enable calculation and display of optional advance post-hoc metrics like perplexity, USE distance, etc.",
        )

        return parser

    @classmethod
    def create_loggers_from_args(cls, args):
        """Creates AttackLogManager from an AttackArgs object."""
        assert isinstance(
            args, cls
        ), f"Expect args to be of type `{type(cls)}`, but got type `{type(args)}`."

        # Create logger
        attack_log_manager = textattack.loggers.AttackLogManager(args.metrics)

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

            color_method = "file"
            attack_log_manager.add_output_file(txt_file_path, color_method)

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

        # if '--log-summary-to-json' specified with arguments
        if args.log_summary_to_json is not None:
            if args.log_summary_to_json.lower().endswith(".json"):
                summary_json_file_path = args.log_summary_to_json
            else:
                summary_json_file_path = os.path.join(
                    args.log_summary_to_json, f"{timestamp}-attack_summary_log.json"
                )

            dir_path = os.path.dirname(summary_json_file_path)
            dir_path = dir_path if dir_path else "."
            if not os.path.exists(dir_path):
                os.makedirs(os.path.dirname(summary_json_file_path))

            attack_log_manager.add_output_summary_json(summary_json_file_path)

        # Visdom
        if args.log_to_visdom is not None:
            attack_log_manager.enable_visdom(**args.log_to_visdom)

        # Weights & Biases
        if args.log_to_wandb is not None:
            attack_log_manager.enable_wandb(**args.log_to_wandb)

        # Stdout
        if not args.disable_stdout and not sys.stdout.isatty():
            attack_log_manager.disable_color()
        elif not args.disable_stdout:
            attack_log_manager.enable_stdout()

        return attack_log_manager


@dataclass
class _CommandLineAttackArgs:
    """Attack args for command line execution.

    This requires more arguments to
    create ``Attack`` object as specified.
    Args:
        transformation (:obj:`str`, `optional`, defaults to :obj:`"word-swap-embedding"`):
            Name of transformation to use.
        constraints (:obj:`list[str]`, `optional`, defaults to :obj:`["repeat", "stopword"]`):
            List of names of constraints to use.
        goal_function (:obj:`str`, `optional`, defaults to :obj:`"untargeted-classification"`):
            Name of goal function to use.
        search_method (:obj:`str`, `optional`, defualts to :obj:`"greedy-word-wir"`):
            Name of search method to use.
        attack_recipe (:obj:`str`, `optional`, defaults to :obj:`None`):
            Name of attack recipe to use.
            .. note::
                Setting this overrides any previous selection of transformation, constraints, goal function, and search method.
        attack_from_file (:obj:`str`, `optional`, defaults to :obj:`None`):
            Path of `.py` file from which to load attack from. Use `<path>^<variable_name>` to specifiy which variable to import from the file.
            .. note::
                If this is set, it overrides any previous selection of transformation, constraints, goal function, and search method
        interactive (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If `True`, carry attack in interactive mode.
        parallel (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If `True`, attack in parallel.
        model_batch_size (:obj:`int`, `optional`, defaults to :obj:`32`):
            The batch size for making queries to the victim model.
        model_cache_size (:obj:`int`, `optional`, defaults to :obj:`2**18`):
            The maximum number of items to keep in the model results cache at once.
        constraint-cache-size (:obj:`int`, `optional`, defaults to :obj:`2**18`):
            The maximum number of items to keep in the constraints cache at once.
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
    model_cache_size: int = 2**18
    constraint_cache_size: int = 2**18

    @classmethod
    def _add_parser_args(cls, parser):
        """Add listed args to command line parser."""
        default_obj = cls()
        transformation_names = set(BLACK_BOX_TRANSFORMATION_CLASS_NAMES.keys()) | set(
            WHITE_BOX_TRANSFORMATION_CLASS_NAMES.keys()
        )
        parser.add_argument(
            "--transformation",
            type=str,
            required=False,
            default=default_obj.transformation,
            help='The transformation to apply. Usage: "--transformation {transformation}:{arg_1}={value_1},{arg_3}={value_3}". Choices: '
            + str(transformation_names),
        )
        parser.add_argument(
            "--constraints",
            type=str,
            required=False,
            nargs="*",
            default=default_obj.constraints,
            help='Constraints to add to the attack. Usage: "--constraints {constraint}:{arg_1}={value_1},{arg_3}={value_3}". Choices: '
            + str(CONSTRAINT_CLASS_NAMES.keys()),
        )
        goal_function_choices = ", ".join(GOAL_FUNCTION_CLASS_NAMES.keys())
        parser.add_argument(
            "--goal-function",
            "-g",
            default=default_obj.goal_function,
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
            default=default_obj.search_method,
            help=f"The search method to use. choices: {search_choices}",
        )
        attack_group.add_argument(
            "--attack-recipe",
            "--recipe",
            "-r",
            type=str,
            required=False,
            default=default_obj.attack_recipe,
            help="full attack recipe (overrides provided goal function, transformation & constraints)",
            choices=ATTACK_RECIPE_NAMES.keys(),
        )
        attack_group.add_argument(
            "--attack-from-file",
            type=str,
            required=False,
            default=default_obj.attack_from_file,
            help="Path of `.py` file from which to load attack from. Use `<path>^<variable_name>` to specifiy which variable to import from the file.",
        )
        parser.add_argument(
            "--interactive",
            action="store_true",
            default=default_obj.interactive,
            help="Whether to run attacks interactively.",
        )
        parser.add_argument(
            "--model-batch-size",
            type=int,
            default=default_obj.model_batch_size,
            help="The batch size for making calls to the model.",
        )
        parser.add_argument(
            "--model-cache-size",
            type=int,
            default=default_obj.model_cache_size,
            help="The maximum number of items to keep in the model results cache at once.",
        )
        parser.add_argument(
            "--constraint-cache-size",
            type=int,
            default=default_obj.constraint_cache_size,
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
    def _create_attack_from_args(cls, args, model_wrapper):
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
            recipe.goal_function.batch_size = args.model_batch_size
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
    def _add_parser_args(cls, parser):
        """Add listed args to command line parser."""
        parser = ModelArgs._add_parser_args(parser)
        parser = DatasetArgs._add_parser_args(parser)
        parser = _CommandLineAttackArgs._add_parser_args(parser)
        parser = AttackArgs._add_parser_args(parser)
        return parser
