from dataclasses import dataclass
import os
import time

import textattack
from textattack.shared.utils import ARGS_SPLIT_TOKEN, load_module_from_file

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
    """Attack args for running attacks via API.

    This assumes that ``Attack`` has already been created by the user.
    Args:
        num_examples (int): The number of examples to attack. -1 for entire dataset.
        num_examples_offset (int): The offset to start at in the dataset.
        attack_n (bool): Whether to run attack until total of `n` examples have been attacked (not skipped).
        checkpoint_dir (str): The directory to save checkpoint files.
        checkpoint_interval (int): If set, checkpoint will be saved after attacking every N examples. If not set, no checkpoints will be saved.
        random_seed (int): Random seed for reproducibility. Default is 765.
        disable_stdout (bool): Disable logging result of each attack to stdout (only produce summary at the end). Default is `False`.
    """

    num_examples: int
    num_examples_offset: int
    attack_n: bool
    checkpoint_dir: str
    checkpoint_interval: int = -1
    random_seed: int = 765  # equivalent to sum((ord(c) for c in "TEXTATTACK"))
    disable_stdout: bool = False

    @classmethod
    def add_parser_args(cls, parser):
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
        parser.add_argument(
            "--attack-n",
            action="store_true",
            default=False,
            help="Whether to run attack until `n` examples have been attacked (not skipped).",
        )

        def default_checkpoint_dir():
            current_dir = os.path.dirname(os.path.realpath(__file__))
            checkpoints_dir = os.path.join(
                current_dir, os.pardir, os.pardir, os.pardir, "checkpoints"
            )
            return os.path.normpath(checkpoints_dir)

        parser.add_argument(
            "--checkpoint-dir",
            required=False,
            type=str,
            default=default_checkpoint_dir(),
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
            "--disable-stdout", action="store_true", help="Disable logging to stdout"
        )
        return parser


@dataclass
class CommandLineAttackArgs(AttackArgs, ModelArgs, DatasetArgs):
    """Command line interface attack args.

    This requires more arguments to create ``Attack`` object as
    specified.
    """

    transformation: str
    constraints: list[str]
    goal_function: str
    search_method: str
    attack_recipe: str
    attack_from_file: str
    interactive: bool = False
    parallel: bool = False
    log_to_text: str = None
    log_to_csv: str = None
    csv_style: str = None
    enable_visdom: bool = False
    enable_wandb: bool = False
    query_budget: int = float("inf")
    model_batch_size: int = 32
    model_cache_size: int = 2 ** 18
    constraint_cache_size: int = 2 ** 18

    @classmethod
    def add_parser_args(cls, parser):
        parser = ModelArgs.add_parser_args(parser)
        parser = DatasetArgs.add_parser_args(parser)

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
            "--search",
            "--search-method",
            "-s",
            type=str,
            required=False,
            default="greedy-word-wir",
            help=f"The search method to use. choices: {search_choices}",
        )
        attack_group.add_argument(
            "--recipe",
            "--attack-recipe",
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
            help="attack to load from file (overrides provided goal function, transformation & constraints)",
        )
        parser.add_argument(
            "--interactive",
            action="store_true",
            default=False,
            help="Whether to run attacks interactively.",
        )
        parser.add_argument(
            "--parallel",
            action="store_true",
            default=False,
            help="Run attack using multiple GPUs.",
        )
        parser.add_argument(
            "--log-to-txt",
            "-l",
            nargs="?",
            default=None,
            const="",
            type=str,
            help="Save attack logs to <install-dir>/outputs/~ by default; Include '/' at the end of argument to save "
            "output to specified directory in default naming convention; otherwise enter argument to specify "
            "file name",
        )
        parser.add_argument(
            "--log-to-csv",
            nargs="?",
            default=None,
            const="",
            type=str,
            help="Save attack logs to <install-dir>/outputs/~ by default; Include '/' at the end of argument to save "
            "output to specified directory in default naming convention; otherwise enter argument to specify "
            "file name",
        )
        parser.add_argument(
            "--csv-style",
            default=None,
            const="fancy",
            nargs="?",
            type=str,
            help="Use --csv-style plain to remove [[]] around words",
        )
        parser.add_argument(
            "--enable-visdom", action="store_true", help="Enable logging to visdom."
        )
        parser.add_argument(
            "--enable-wandb",
            action="store_true",
            help="Enable logging to Weights & Biases.",
        )
        parser.add_argument(
            "--query-budget",
            "-q",
            type=int,
            default=float("inf"),
            help="The maximum number of model queries allowed per example attacked.",
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

        parser = AttackArgs.add_parser_args(parser)

        return parser

    @classmethod
    def _parse_transformation_from_args(cls, args, model_wrapper):
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
    def _parse_goal_function_from_args(cls, args, model_wrapper):
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
        goal_function.query_budget = args.query_budget
        goal_function.model_batch_size = args.model_batch_size
        goal_function.model_cache_size = args.model_cache_size
        return goal_function

    @classmethod
    def _parse_constraints_from_args(cls, args):
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
    def parse_attack_from_args(cls, args, model_wrapper):
        assert isinstance(
            args, cls
        ), f"Expect args to be of type `{type(cls)}`, but got type `{type(args)}`."

        if args.recipe:
            if ARGS_SPLIT_TOKEN in args.recipe:
                recipe_name, params = args.recipe.split(ARGS_SPLIT_TOKEN)
                if recipe_name not in ATTACK_RECIPE_NAMES:
                    raise ValueError(f"Error: unsupported recipe {recipe_name}")
                recipe = eval(
                    f"{ATTACK_RECIPE_NAMES[recipe_name]}.build(model_wrapper, {params})"
                )
            elif args.recipe in ATTACK_RECIPE_NAMES:
                recipe = eval(
                    f"{ATTACK_RECIPE_NAMES[args.recipe]}.build(model_wrapper)"
                )
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
            return attack_func(model_wrapper)
        else:
            goal_function = cls._parse_goal_function_from_args(args, model_wrapper)
            transformation = cls._parse_transformation_from_args(args, model_wrapper)
            constraints = cls._parse_constraints_from_args(args)
            if ARGS_SPLIT_TOKEN in args.search:
                search_name, params = args.search.split(ARGS_SPLIT_TOKEN)
                if search_name not in SEARCH_METHOD_CLASS_NAMES:
                    raise ValueError(f"Error: unsupported search {search_name}")
                search_method = eval(
                    f"{SEARCH_METHOD_CLASS_NAMES[search_name]}({params})"
                )
            elif args.search in SEARCH_METHOD_CLASS_NAMES:
                search_method = eval(f"{SEARCH_METHOD_CLASS_NAMES[args.search]}()")
            else:
                raise ValueError(f"Error: unsupported attack {args.search}")
        return textattack.Attack(
            goal_function,
            constraints,
            transformation,
            search_method,
            constraint_cache_size=args.constraint_cache_size,
        )

    @classmethod
    def parse_logger_from_args(cls, args):
        assert isinstance(
            args, cls
        ), f"Expect args to be of type `{type(cls)}`, but got type `{type(args)}`."

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
