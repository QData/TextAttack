import os
from attack_args_helper import HUGGINGFACE_DATASET_BY_MODEL

dir_path = os.path.dirname(os.path.realpath(__file__))
for model in HUGGINGFACE_DATASET_BY_MODEL:
    if not model.startswith('bert'):
        os.system(f'python {os.path.join(dir_path, "benchmark_models.py")} --model {model} --num-examples 200')