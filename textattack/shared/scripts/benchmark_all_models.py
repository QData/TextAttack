import os

from textattack.shared.scripts.attack_args import (
    HUGGINGFACE_DATASET_BY_MODEL,
    TEXTATTACK_DATASET_BY_MODEL,
)

if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))
    for model in {**TEXTATTACK_DATASET_BY_MODEL, **HUGGINGFACE_DATASET_BY_MODEL}:
        print(model)
        os.system(
            f'python {os.path.join(dir_path, "benchmark_model.py")} --model {model} --num-examples 1000'
        )
        print()
