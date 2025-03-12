import textattack
from textattack import Attack

import datasets
from datasets import load_from_disk

import json
import os
import time

def toxic_attack(goal_function, constraints, transformation, search_method, model_wrapper, valid_dataset_path, num_rows, results_path):

    attack = Attack(goal_function, constraints, transformation, search_method)

    with open(valid_dataset_path, "r", encoding="utf-8") as f:
        valid_dataset = json.load(f)

        with open(results_path, "w", encoding="utf8") as results:
            index = 0
            for tc in valid_dataset[:num_rows]:
                input_text = tc["comment"]
                start_time = time.time()
                attack_result = attack.attack(input_text, 1)
                elapsed_time = time.time() - start_time
                result_entry = {
                    "index": index,
                    "elapsed_time": elapsed_time,
                    "input_text": input_text,
                    "perturbed_text": attack_result.attacked_text.text,
                    "perturbed_output": float(attack_result.output['toxic']),
                    "toxic_sum": float(attack_result.score),
                    "goal_status": attack_result.goal_status
                }
                print(result_entry)
                results.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                index += 1
                print(f"Processed: {index} / {num_rows}")
