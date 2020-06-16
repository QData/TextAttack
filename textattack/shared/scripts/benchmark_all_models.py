import os
from attack_args_helper import TEXTATTACK_MODEL_CLASS_NAMES, HUGGINGFACE_DATASET_BY_MODEL

dir_path = os.path.dirname(os.path.realpath(__file__))
for model in {**TEXTATTACK_MODEL_CLASS_NAMES, **HUGGINGFACE_DATASET_BY_MODEL}:
    print(model)
    os.system(f'python {os.path.join(dir_path, "benchmark_models.py")} --model {model} --num-examples 1000')
    print()