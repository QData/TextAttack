import textattack
from textattack import Attack

import datasets
from datasets import load_from_disk

import json
import os
import time

from collections import OrderedDict

def mnli_targeted_attack(goal_function, constraints, transformation, search_method, valid_dataset_path, num_rows, results_path):

    attack = Attack(goal_function, constraints, transformation, search_method)

    label_map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

    with open(valid_dataset_path, 'r') as f:
        mnli_test = []
        for jline in f.readlines():
            line = json.loads(jline)
            if line['gold_label'] in label_map:
                mnli_test.append(line)

    with open(results_path, "a", encoding="utf8") as results:
        index = 0
        for tc in mnli_test[:num_rows]:
            for target in range(0, 3):
                inp = OrderedDict([('premise', tc['sentence1']), ('hypothesis', tc['sentence2'])])
                start_time = time.time()
                attack_result = attack.attack(inp, target).perturbed_result
                print(attack_result)
                elapsed_time = time.time() - start_time
                result_entry = {
                    "index": index,
                    "elapsed_time": elapsed_time,
                    "input_text": inp,
                    "target": target,
                    "perturbed_text": attack_result.attacked_text.text,
                    "perturbed_output": attack_result.output,
                    "score": attack_result.score,
                    "goal_status": attack_result.goal_status
                }
                results.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                index += 1
                print(f"Processed: {index} / {num_rows}")
    