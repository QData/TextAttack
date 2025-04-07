import textattack
from textattack import Attack

import datasets
from datasets import load_from_disk

import json
import os
import time

from helper import ner_tags, detokenize

def ner_targeted_attack(goal_function, constraints, transformation, search_method, valid_dataset_path, num_rows, results_path):

    attack = Attack(goal_function, constraints, transformation, search_method)

    valid_dataset = load_from_disk(valid_dataset_path)

    with open(results_path, "a", encoding="utf8") as results:
        index = 0
        for tc in valid_dataset.select(range(num_rows)):

            ner_classes = ['PER', 'ORG', 'LOC', 'MISC']

            tokens = tc['tokens']
            inp = detokenize(tokens)
            labels = ner_tags(tc['ner_tags'])

            target = ner_classes[0]

            start_time = time.time()
            attack_result = attack.attack(inp, target)
            print(attack_result)
            elapsed_time = time.time() - start_time
            result_entry = {
                "index": index,
                "elapsed_time": elapsed_time,
                "input_text": inp,
                # "correct_output": labels,
                "target": target,
                "perturbed_text": attack_result.attacked_text.text,
                # "perturbed_output": attack_result.output,
                "score": attack_result.score,
                "goal_status": attack_result.goal_status
            }
            results.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            index += 1
            print(f"Processed: {index} / {num_rows}")
    