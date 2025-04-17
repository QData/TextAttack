import textattack

from textattack.goal_functions import Ner
from textattack.search_methods import ImperceptibleDE
from textattack.transformations import WordSwapHomoglyphSwap
from load import load_ner_from_local_cache, load_ner_data_from_local_cache, load_ner
from helper import detokenize, ner_tags
from att import ner_targeted_attack

import transformers
from transformers import AutoTokenizer, FSMTModel, FSMTForConditionalGeneration

import datasets
from datasets import load_dataset

import os
import datetime

# Model wrapper

cur_dir = os.path.dirname(os.path.abspath(__file__))

# model_path = os.path.join(cur_dir, 'local_ner_model')
# tokenizer_path = os.path.join(cur_dir, 'local_ner_tokenizer')
data_path = os.path.join(cur_dir, 'ner_data_all')
 
# model = load_ner_from_local_cache(model_path, tokenizer_path)
model = load_ner()
model_wrapper = textattack.models.wrappers.PipelineModelWrapper(model)

data = load_ner_data_from_local_cache(data_path)

# test = data[0]
# inp = detokenize(test['tokens'])
# predicts = model(inp)
# labels = ner_tags(test['ner_tags'])

# print(inp)
# print(predicts)
# print(test)
# print(labels)

# Attack params
popsize = 10
maxiter = 5
max_perturbs = 2

goal_function = Ner(model_wrapper)
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

results_path = os.path.join(results_dir, f"ner_targeted_homo_pop{popsize}_iter{maxiter}_perturbs{max_perturbs}.jsonl")


ner_targeted_attack(
    goal_function=goal_function, 
    constraints=constraints, 
    transformation=transformation, 
    search_method=search_method,
    valid_dataset_path=data_path, 
    num_rows=10,
    results_path=results_path
)
