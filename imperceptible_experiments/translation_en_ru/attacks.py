import textattack
from textattack import Attack

import datasets
from datasets import load_from_disk

import json
import os
import time

def translation_attack_en_ru(goal_function, constraints, transformation, search_method, model_wrapper, valid_dataset_path, num_rows, results_path):

    attack = Attack(goal_function, constraints, transformation, search_method)

    valid_dataset = load_from_disk(valid_dataset_path)

    with open(results_path, "a", encoding="utf8") as results:
        index = 0
        for tc in valid_dataset.select(range(num_rows)):
            input_text = tc["translation"]["en"]
            correct_translation = tc["translation"]["ru"]
            start_time = time.time()
            attack_result = attack.attack(input_text, correct_translation)
            elapsed_time = time.time() - start_time
            result_entry = {
                "index": index,
                "elapsed_time": elapsed_time,
                "input_text": input_text,
                "correct_translation": correct_translation,
                "perturbed_text": attack_result.attacked_text.text,
                "perturbed_output": attack_result.output,
                "lev_score": attack_result.score,
                "goal_status": attack_result.goal_status
            }
            results.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            index += 1
            print(f"Processed: {index} / {num_rows}")
    