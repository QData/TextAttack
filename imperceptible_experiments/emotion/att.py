import textattack
from textattack import Attack

import datasets
from datasets import load_from_disk

import json
import os
import time

def emotion_attack(goal_function, constraints, transformation, search_method, valid_dataset_path, num_rows, results_path):

    attack = Attack(goal_function, constraints, transformation, search_method)

    valid_dataset = load_from_disk(valid_dataset_path)

    emotion_classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    with open(results_path, "a", encoding="utf8") as results:
        index = 0
        for tc in valid_dataset.select(range(num_rows)):
            input_text = tc['text']
            model_pred = tc['label']
            model_pred_label = emotion_classes[model_pred]
            target = 0
            target_label = emotion_classes[target]
            start_time = time.time()
            attack_result = attack.attack(input_text, target)
            elapsed_time = time.time() - start_time
            result_entry = {
                "index": index,
                "elapsed_time": elapsed_time,
                "input_text": input_text,
                "model_pred_label": model_pred_label,
                "target_label": target_label,
                "perturbed_text": attack_result.attacked_text.text,
                "perturbed_output": attack_result.output,
                "score": attack_result.score,
                "goal_status": attack_result.goal_status
            }
            results.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            index += 1
            print(f"Processed: {index} / {num_rows}")
    