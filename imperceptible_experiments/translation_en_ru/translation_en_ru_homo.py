import textattack

from imperceptible_experiments.imperceptible_attacks import translation_attack_en_ru
from textattack.goal_functions import LevenshteinExceedsTargetDistance
from textattack.search_methods import DifferentialEvolutionSearchHomoglyph
from textattack.transformations import WordSwapHomoglyphSwap

import transformers
from transformers import AutoTokenizer, FSMTModel, FSMTForConditionalGeneration

import datasets
from datasets import load_dataset

# Model wrapper
model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-ru")
tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-en-ru")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapperLang(model, tokenizer)

# Load validation dataset
valid_dataset = load_dataset("wmt/wmt19", "ru-en", split="validation")
valid_dataset.save_to_disk("wmt19_ru_en_validation")
# valid_dataset.to_json("wmt19_ru_en_validation.json")

# Attack params
goal_function = LevenshteinExceedsTargetDistance(model_wrapper)
constraints = []
transformation = WordSwapHomoglyphSwap()
search_method = DifferentialEvolutionSearchHomoglyph(
    popsize=10, 
    maxiter=5, 
    verbose=True,
    max_perturbs=1
)

translation_attack_en_ru(
    goal_function=goal_function, 
    constraints=constraints, 
    transformation=transformation, 
    search_method=search_method,
    model_wrapper=model_wrapper,
    valid_dataset_path="wmt19_ru_en_validation", 
    num_rows=10,
    results_path="translation_homo_small.jsonl"
)
