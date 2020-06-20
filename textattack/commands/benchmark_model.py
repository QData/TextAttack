from argparse import ArgumentParser

import argparse
import collections
import sys

import torch

# from attack_args_parser import get_args, parse_dataset_from_args, parse_model_from_args
import textattack

from textattack.commands import TextAttackCommand
class BenchmarkModelCommand(TextAttackCommand):
    """
    The TextAttack model benchmarking module:
    
        A command line parser to benchmark a model from user specifications.
    """
    
    def run(self):
        raise NotImplementedError('cant benchmark yet')

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser("benchmark-model", help="evaluate a model with TextAttack")
        parser.set_defaults(func=BenchmarkModelCommand())
        
def _cb(s):
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")


def get_num_successes(args, model, ids, true_labels):
    with torch.no_grad():
        preds = textattack.shared.utils.model_predict(model, ids)
    true_labels = torch.tensor(true_labels).to(textattack.shared.utils.device)
    guess_labels = preds.argmax(dim=1)
    successes = (guess_labels == true_labels).sum().item()
    return successes, true_labels, guess_labels


def test_model_on_dataset(args, model, dataset, num_examples=100, batch_size=128):
    num_examples = args.num_examples
    succ = 0
    fail = 0
    batch_ids = []
    batch_labels = []
    all_true_labels = []
    all_guess_labels = []
    for i, (text_input, label) in enumerate(dataset):
        if i >= num_examples:
            break
        attacked_text = textattack.shared.AttackedText(text_input)
        ids = model.tokenizer.encode(attacked_text.tokenizer_input)
        batch_ids.append(ids)
        batch_labels.append(label)
        if len(batch_ids) == batch_size:
            batch_succ, true_labels, guess_labels = get_num_successes(
                args, model, batch_ids, batch_labels
            )
            batch_fail = batch_size - batch_succ
            succ += batch_succ
            fail += batch_fail
            batch_ids = []
            batch_labels = []
            all_true_labels.extend(true_labels.tolist())
            all_guess_labels.extend(guess_labels.tolist())
    if len(batch_ids) > 0:
        batch_succ, true_labels, guess_labels = get_num_successes(
            args, model, batch_ids, batch_labels
        )
        batch_fail = len(batch_ids) - batch_succ
        succ += batch_succ
        fail += batch_fail
        all_true_labels.extend(true_labels.tolist())
        all_guess_labels.extend(guess_labels.tolist())

    perc = float(succ) / (succ + fail) * 100.0
    perc = "{:.2f}%".format(perc)
    print(f"Successes {succ}/{succ+fail} ({_cb(perc)})")
    return perc


if __name__ == "__main__":
    args = get_args()

    model = parse_model_from_args(args)
    dataset = parse_dataset_from_args(args)

    with torch.no_grad():
        test_model_on_dataset(args, model, dataset, num_examples=args.num_examples)


""" Old benchmark_all_models """
# import os
# from textattack.shared.scripts.attack_args import (
#     HUGGINGFACE_DATASET_BY_MODEL,
#     TEXTATTACK_DATASET_BY_MODEL,
# )

# if __name__ == "__main__":

#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     for model in {**TEXTATTACK_DATASET_BY_MODEL, **HUGGINGFACE_DATASET_BY_MODEL}:
#         print(model)
#         os.system(
#             f'python {os.path.join(dir_path, "benchmark_model.py")} --model {model} --num-examples 1000'
#         )
#         print()
