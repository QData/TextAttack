import textattack
from textattack import Attack

import datasets
from datasets import load_from_disk

import json
import os
import time

import multiprocessing as mp

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



def load_dataset(path, num_rows):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:num_rows]

def run_single_attack(index, input_text, attack):
    try:
        start = time.time()
        result = attack.attack(input_text, 1)
        elapsed = time.time() - start

        return {
            "index": index,
            "elapsed_time": elapsed,
            "input_text": input_text,
            "perturbed_text": result.attacked_text.text,
            "perturbed_output": float(result.output["toxic"]),
            "toxic_sum": float(result.score),
            "goal_status": result.goal_status
        }
    except Exception as e:
        return {
            "index": index,
            "input_text": input_text,
            "error": str(e)
        }

def toxic_attack_parallel(goal_function, constraints, transformation, search_method, model_wrapper, valid_dataset_path, num_rows, results_path):
    # Build the attack object ONCE
    attack = Attack(goal_function, constraints, transformation, search_method)

    # Load dataset
    dataset = load_dataset(valid_dataset_path, num_rows)
    print(f"Loaded {len(dataset)} rows")

    # Prep tasks
    tasks = [(i, row["comment"], attack) for i, row in enumerate(dataset)]

    # Force torch to behave well
    import torch
    torch.set_num_threads(1)

    # Run in parallel
    num_workers = mp.cpu_count()
    print(f"Using {num_workers} cores")

    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(run_single_attack, tasks)

    # Save to .jsonl
    with open(results_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved results to: {results_path}")
