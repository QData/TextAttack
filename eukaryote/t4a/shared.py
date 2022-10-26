import json

import eukaryote
from eukaryote.datasets import HuggingFaceDataset
from eukaryote.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# all attacks that can be used, all from original eukaryote
attack_recipe_names = {
    "alzantot": eukaryote.attack_recipes.GeneticAlgorithmAlzantot2018,
    "bae": eukaryote.attack_recipes.BAEGarg2019,
    "bert-attack": eukaryote.attack_recipes.BERTAttackLi2020,
    "faster-alzantot": eukaryote.attack_recipes.FasterGeneticAlgorithmJia2019,
    "deepwordbug": eukaryote.attack_recipes.DeepWordBugGao2018,
    "hotflip": eukaryote.attack_recipes.HotFlipEbrahimi2017,
    "input-reduction": eukaryote.attack_recipes.InputReductionFeng2018,
    "kuleshov": eukaryote.attack_recipes.Kuleshov2017,
    "morpheus": eukaryote.attack_recipes.MorpheusTan2020,
    "seq2sick": eukaryote.attack_recipes.Seq2SickCheng2018BlackBox,
    "textbugger": eukaryote.attack_recipes.TextBuggerLi2018,
    "textfooler": eukaryote.attack_recipes.TextFoolerJin2019,
    "pwws": eukaryote.attack_recipes.PWWSRen2019,
    "iga": eukaryote.attack_recipes.IGAWang2019,
    "pruthi": eukaryote.attack_recipes.Pruthi2019,
    "pso": eukaryote.attack_recipes.PSOZang2020,
    "checklist": eukaryote.attack_recipes.CheckList2020,
    "clare": eukaryote.attack_recipes.CLARE2020,
    "a2t": eukaryote.attack_recipes.A2TYoo2021,
}

constraint_class_names = {
    "embedding": eukaryote.constraints.semantics.WordEmbeddingDistance,
    "bert": eukaryote.constraints.semantics.sentence_encoders.BERT,
    "infer-sent": eukaryote.constraints.semantics.sentence_encoders.InferSent,
    "thought-vector": eukaryote.constraints.semantics.sentence_encoders.ThoughtVector,
    "use": eukaryote.constraints.semantics.sentence_encoders.UniversalSentenceEncoder,
    "muse": eukaryote.constraints.semantics.sentence_encoders.MultilingualUniversalSentenceEncoder,
    "bert-score": eukaryote.constraints.semantics.BERTScore,
    "lang-tool": eukaryote.constraints.grammaticality.LanguageTool,
    "part-of-speech": eukaryote.constraints.grammaticality.PartOfSpeech,
    "goog-lm": eukaryote.constraints.grammaticality.language_models.Google1BillionWordsLanguageModel,
    "gpt2": eukaryote.constraints.grammaticality.language_models.GPT2,
    "learning-to-write": eukaryote.constraints.grammaticality.language_models.LearningToWriteLanguageModel,
    "cola": eukaryote.constraints.grammaticality.COLA,
    "bleu": eukaryote.constraints.overlap.BLEU,
    "chrf": eukaryote.constraints.overlap.chrF,
    "edit-distance": eukaryote.constraints.overlap.LevenshteinEditDistance,
    "meteor": eukaryote.constraints.overlap.METEOR,
    "max-words-perturbed": eukaryote.constraints.overlap.MaxWordsPerturbed,
    "repeat": eukaryote.constraints.pre_transformation.RepeatModification,
    "stopword": eukaryote.constraints.pre_transformation.StopwordModification,
    "max-modification-rate": eukaryote.constraints.pre_transformation.MaxModificationRate,
    "max-word-index": eukaryote.constraints.pre_transformation.MaxWordIndexModification,
}


def nullable(type_fn):
    """Helper function that wraps a type constructor to create nullable
    types in `argparse` arguments.
    """

    def fn(x):
        if x in ["null", "None"]:
            return None
        return type_fn(x)

    return fn


def add_arguments_model(parser):
    """Define model arguments."""
    group = parser.add_argument_group("Model")
    group_mutex = group.add_mutually_exclusive_group(required=True)
    group_mutex.add_argument(
        "--model_huggingface",
        type=str,
        help="Name or path of HuggingFace model",
    )
    group_mutex.add_argument(
        "--model_tensorflow",
        type=str,
        help="Name of TensorFlow model",
    )


def add_arguments_dataset(parser, default_split=None):
    """Define dataset arguments."""
    group = parser.add_argument_group("Dataset")
    group_mutex = group.add_mutually_exclusive_group(required=True)
    group_mutex.add_argument(
        "--dataset_huggingface",
        type=str,
        help="Name of HuggingFace dataset",
    )
    group_mutex.add_argument(
        "--dataset_csv",
        type=str,
        help="Path to csv dataset",
    )
    group.add_argument(
        "--dataset_split",
        type=str,
        default=default_split,
        help="Name of dataset split",
    )


def add_arguments_attack(
    parser, enable_attack_args=True, enable_extra_constraints=True
):
    """Define attack arguments."""
    group = parser.add_argument_group("Attack")
    group.add_argument(
        "--attack_recipe",
        type=str,
        required=True,
        choices=attack_recipe_names.keys(),
        help="Name of attack recipe",
    )
    if enable_attack_args:
        group.add_argument(
            "--attack_args",
            type=str,
            help="JSON string of extra kwargs to pass to the attacker",
        )
    if enable_extra_constraints:
        group.add_argument(
            "--perturbation_budget_class",
            default="max-modification-rate",
            choices=constraint_class_names.keys(),
            help="Name of perturbation budget class",
        )
        group_mutex = group.add_mutually_exclusive_group()
        group_mutex.add_argument(
            "--perturbation_budget",
            type=nullable(float),
            nargs="+",
            help="Values to optionally constraint perturbation budget",
        )
        group_mutex.add_argument(
            "--attack_strength",
            type=nullable(int),
            nargs="+",
            help="Values to optionally constrain attack strength",
        )


def add_arguments_train(parser, default_split_eval=None):
    """Define training arguments."""
    group = parser.add_argument_group("Train")
    group.add_argument(
        "--save",
        type=str,
        help="Path to save model",
    )
    group.add_argument(
        "--dataset_split_eval",
        type=str,
        default=default_split_eval,
        help="Name of dataset split for validation while training",
    )
    group.add_argument("--epochs", type=int, help="Stop training after n epochs")
    group.add_argument(
        "--early_stopping_epochs",
        type=int,
        help="Stop training when validation increases for m epochs",
    )
    group.add_argument("--learning_rate", type=float, help="Learning rate")
    group.add_argument("--batch_size", type=int, help="Batch size (per device)")


def load_model_wrapper(args):
    """Load and return a eukaryote `ModelWrapper` depending on the
    arguments.

    TODO: This fails on certain named Hugging Face models which require some
    configuration to be set beyond what `AutoModel...` and `AutoTokenizer` can
    determine in order to be compatible with eukaryote.
    """
    if args.model_huggingface:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_huggingface
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_huggingface)
        return HuggingFaceModelWrapper(model, tokenizer)
    if args.model_tensorflow:
        # TODO: Add loading TensorFlow models
        raise NotImplementedError


def load_dataset(args, split=None):
    """Load and return a eukaryote `Dataset` depending on the arguments."""
    if split is None:
        split = args.dataset_split
    if args.dataset_huggingface:
        return HuggingFaceDataset(args.dataset_huggingface, split=split)
    if args.dataset_csv:
        # TODO: Add loading datasets from a csv file
        raise NotImplementedError


def load_attack(args):
    """Load an attack and related arguments.

    Returns:
        dict: dictionary containing:

            - `attack_recipe` (eukaryote.attack_recipes.AttackRecipe):
                The attack recipe.
            - `perturbation_budget_class` (Optional[Union[
                eukaryote.constraints.PreTransformationConstraint,
                eukaryote.constraints.Constraint,
                ]]):
                The class of the constraint which defines perturbation budget.
            - `perturbation_budget` (Optional[list]):
                A (possibly singleton) list of perturbation budget values.
            - `attack_strength` (Optional[list[int]]):
                A (possibly singleton) list of attack strength values.
            - `attack_args` (Optional[dict]):
                A dict of extra kwargs to pass to the attacker.
    """

    result = {}

    if args.attack_recipe not in attack_recipe_names:
        raise ValueError("Unknown attack recipe name")
    result["attack_recipe"] = attack_recipe_names[args.attack_recipe]

    if args.perturbation_budget:
        if args.perturbation_budget_class not in constraint_class_names:
            raise ValueError("Unknown perturbation budget class name")
        result["perturbation_budget_class"] = constraint_class_names[
            args.perturbation_budget_class
        ]
        result["perturbation_budget"] = args.perturbation_budget

    if args.attack_strength:
        result["attack_strength"] = args.attack_strength

    if args.attack_args:
        result["attack_args"] = json.loads(args.attack_args)

    return result
