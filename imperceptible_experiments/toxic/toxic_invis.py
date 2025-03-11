import textattack
from attacks import toxic_attack
from textattack.goal_functions import Toxic
from textattack.search_methods import DifferentialEvolutionSearchHomoglyph, DifferentialEvolutionSearchInvisibleChars
from textattack.transformations import WordSwapHomoglyphSwap, WordSwapInvisibleCharacters

import transformers
from transformers import AutoTokenizer, FSMTModel, FSMTForConditionalGeneration

import datasets
from datasets import load_dataset

import json

from imperceptible_experiments.toxic.toxic.config import DEFAULT_MODEL_PATH, LABEL_LIST, MODEL_META_DATA as model_meta
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
from imperceptible_experiments.toxic.toxic.core.bert_pytorch import BertForMultiLabelSequenceClassification

import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

# print("Current working directory:", os.getcwd())

# Load Model
tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_PATH, do_lower_case=True)
model_state_dict = torch.load(DEFAULT_MODEL_PATH + "pytorch_model.bin", map_location='cpu')
model = BertForMultiLabelSequenceClassification.from_pretrained(DEFAULT_MODEL_PATH,
                                                                             num_labels=len(LABEL_LIST),
                                                                             state_dict=model_state_dict)
model_wrapper = textattack.models.wrappers.IBMBertModelWrapper(model, tokenizer)

# Attack params
goal_function = Toxic(model_wrapper)
constraints = []
popsize = 32
maxiter = 5

if __name__ == "__main__":

    valid_dataset_path = os.path.join(cur_dir, "toxic_test.json")

    for max_perturbs in range(3, 4): 
        transformation = WordSwapInvisibleCharacters()
        search_method = DifferentialEvolutionSearchInvisibleChars(
            popsize=popsize, 
            maxiter=maxiter, 
            verbose=True,
            max_perturbs=max_perturbs
        )

        results_path = os.path.join(cur_dir, f"results/toxic_invis_pop{popsize}_iter{maxiter}_perturbs{max_perturbs}.jsonl")

        print(f"Running attack with max_perturbs={max_perturbs}... Output: {results_path}")

        toxic_attack(
            goal_function=goal_function, 
            constraints=constraints, 
            transformation=transformation, 
            search_method=search_method,
            model_wrapper=model_wrapper,
            valid_dataset_path=valid_dataset_path, 
            num_rows=5,
            results_path=results_path
        )

    print("Finished running all attacks!")
