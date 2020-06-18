import os

<<<<<<< HEAD
from attack_args_parser import (
=======
from attack_args_helper import (
>>>>>>> 6953f0ee7d024957774d19d101175f0fa0176ccc
    HUGGINGFACE_DATASET_BY_MODEL,
    TEXTATTACK_MODEL_CLASS_NAMES,
)

dir_path = os.path.dirname(os.path.realpath(__file__))
for model in {**TEXTATTACK_MODEL_CLASS_NAMES, **HUGGINGFACE_DATASET_BY_MODEL}:
    print(model)
    os.system(
        f'python {os.path.join(dir_path, "benchmark_models.py")} --model {model} --num-examples 1000'
    )
    print()
