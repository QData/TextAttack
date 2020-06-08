import os
from attack_args_helper import HUGGINGFACE_DATASET_BY_MODEL


for model in HUGGINGFACE_DATASET_BY_MODEL:
    if model.startswith('x'):
        os.system(f'python benchmark_models.py --model {model} --num-examples 200')