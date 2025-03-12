import textattack
from att import toxic_attack
from textattack.goal_functions import Toxic
from textattack.search_methods import ImperceptibleDE
from textattack.transformations import WordSwapInvisibleCharacters, WordSwapHomoglyphSwap, WordSwapDeletions, WordSwapReorderings

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
import datetime

cur_dir = os.path.dirname(os.path.abspath(__file__))

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
popsize = 10
maxiter = 5

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

to_run = [False, False, False, True] # invis, reorderings, homoglyphs, deletions

if __name__ == "__main__":

    valid_dataset_path = os.path.join(cur_dir, "toxic_test.json")

    results_dir = os.path.join(cur_dir, "results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # invis
    if (to_run[0]):
        for max_perturbs in range(1, 2): 
            transformation = WordSwapInvisibleCharacters()
            search_method = ImperceptibleDE(
                popsize=popsize, 
                maxiter=maxiter, 
                verbose=True,
                max_perturbs=max_perturbs
            )

            results_path = os.path.join(results_dir, f"toxic_invis_pop{popsize}_iter{maxiter}_perturbs{max_perturbs}.jsonl")

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

    # reorderings
    if (to_run[1]):
        for max_perturbs in range(1, 2): 
            transformation = WordSwapReorderings()
            search_method = ImperceptibleDE(
                popsize=popsize, 
                maxiter=maxiter, 
                verbose=True,
                max_perturbs=max_perturbs
            )

            results_path = os.path.join(results_dir, f"toxic_reorderings_pop{popsize}_iter{maxiter}_perturbs{max_perturbs}.jsonl")

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

    # homoglyphs
    if (to_run[2]):
        for max_perturbs in range(1, 2): 
            transformation = WordSwapHomoglyphSwap()
            search_method = ImperceptibleDE(
                popsize=popsize, 
                maxiter=maxiter, 
                verbose=True,
                max_perturbs=max_perturbs
            )
            
            results_path = os.path.join(results_dir, f"toxic_homoglyphs_pop{popsize}_iter{maxiter}_perturbs{max_perturbs}.jsonl")

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

    # deletions
    if (to_run[3]):
        for max_perturbs in range(1, 2): 
            transformation = WordSwapDeletions()
            search_method = ImperceptibleDE(
                popsize=popsize, 
                maxiter=maxiter, 
                verbose=True,
                max_perturbs=max_perturbs
            )

            results_path = os.path.join(results_dir, f"toxic_deletions_pop{popsize}_iter{maxiter}_perturbs{max_perturbs}.jsonl")

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
