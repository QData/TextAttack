import textattack
from textattack import Attack

import datasets
from datasets import load_from_disk

import json
import os
import time

def translation_attack(goal_function, constraints, transformation, search_method, model_wrapper, source, target, num_rows, results_path):

    attack = Attack(goal_function, constraints, transformation, search_method)

    with open(results_path, "a", encoding="utf8") as results:
        index = 0
        for i, example in enumerate(source): # example has format {docid: {segid: sec}}
            for docid, doc in example.items(): # only one (key, value) pair
                for segid, sec in doc.items(): # only one (key, value) pair
                    ref = target[i][docid][segid]
                    start_time = time.time()
                    attack_result = attack.attack(sec, ref).perturbed_result
                    print(attack_result)
                    elapsed_time = time.time() - start_time
                    result_entry = {
                        "index": index,
                        "elapsed_time": elapsed_time,
                        "input_text": sec,
                        "correct_translation": ref,
                        "perturbed_text": attack_result.attacked_text.text,
                        "perturbed_output": attack_result.output,
                        "score": attack_result.score,
                        "goal_status": attack_result.goal_status
                    }
                    results.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                    index += 1
                    print(f"Processed: {index} / {num_rows}")
    