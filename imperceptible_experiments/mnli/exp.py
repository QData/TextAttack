import textattack

from textattack.goal_functions import Mnli
from textattack.search_methods import ImperceptibleDE
from textattack.transformations import WordSwapHomoglyphSwap
from att import mnli_targeted_attack

import transformers
from transformers import AutoTokenizer, FSMTModel, FSMTForConditionalGeneration

import datasets
from datasets import load_dataset

import os
import datetime

import torch

# Model wrapper
 
model = torch.hub.load('pytorch/fairseq',
                       'roberta.large.mnli').eval()

model_wrapper = textattack.models.wrappers.FairseqMnliWrapper(model)


# Attack params
popsize = 10
maxiter = 5
max_perturbs = 1

goal_function = Mnli(model_wrapper)
constraints = []
transformation = WordSwapHomoglyphSwap()
search_method = ImperceptibleDE(
    popsize=popsize, 
    maxiter=maxiter, 
    verbose=True,
    max_perturbs=max_perturbs
)



cur_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join(cur_dir, "results", timestamp)
os.makedirs(results_dir, exist_ok=True)

results_path = os.path.join(results_dir, f"mnli_homo_pop{popsize}_iter{maxiter}_perturbs{max_perturbs}.jsonl")

data_path = os.path.join(cur_dir, 'multinli_1.0/multinli_1.0_dev_matched.jsonl')


mnli_targeted_attack(
    goal_function=goal_function, 
    constraints=constraints, 
    transformation=transformation, 
    search_method=search_method,
    valid_dataset_path=data_path, 
    num_rows=10,
    results_path=results_path
)
