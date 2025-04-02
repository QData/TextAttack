import textattack

from textattack.goal_functions import Emotion
from textattack.search_methods import ImperceptibleDE
from textattack.transformations import WordSwapHomoglyphSwap
from imperceptible_experiments.emotion.load import load_emotion_from_local_cache
from att import emotion_attack

import transformers
from transformers import AutoTokenizer, FSMTModel, FSMTForConditionalGeneration

import datasets
from datasets import load_dataset

import os
import datetime

# Model wrapper

model_path = './local_emotion_model'
tokenizer_path = './local_emotion_tokenizer'
 
model = load_emotion_from_local_cache(model_path, tokenizer_path)

model_wrapper = textattack.models.wrappers.PipelineModelWrapper(model)


# Attack params
popsize = 10
maxiter = 5
max_perturbs = 2

goal_function = Emotion(model_wrapper)
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

results_path = os.path.join(results_dir, f"emotion_homo_pop{popsize}_iter{maxiter}_perturbs{max_perturbs}.jsonl")


emotion_attack(
    goal_function=goal_function, 
    constraints=constraints, 
    transformation=transformation, 
    search_method=search_method,
    valid_dataset_path="emotion_data_all", 
    num_rows=10,
    results_path=results_path
)
