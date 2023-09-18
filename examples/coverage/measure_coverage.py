"""

python measure_coverage.py --split test --ratio 1.0 --coverage attention --length 128 --prespecify-limits  --attack textfooler --test-ratio 1000 --save-dir ./coverage_wp/ --seed 1 --dataset sst2


"""

import argparse
from math import floor
import os
import pickle
import random

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

import textattack
from textattack.attack_recipes import (
    BAEGarg2019,
    DeepWordBugGao2018,
    FasterGeneticAlgorithmJia2019,
    HotFlipEbrahimi2017,
    TextFoolerJin2019,
)
from textattack.attack_results import SuccessfulAttackResult
from textattack.coverage import neuronMultiSectionCoverage
from textattack.datasets import HuggingFaceDataset
from textattack.models.tokenizers import AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper, ModelWrapper


def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return


random_seed(1)
parser = argparse.ArgumentParser(
    description="Measure Coverage of pretrained NLP Models"
)
parser.add_argument("--seed", type=int, default=1, help="set random seed")
parser.add_argument("--length", type=int, default=128, help="set max seq length")
parser.add_argument("--bins", type=int, default=10, help="set number of	bins/sections")
parser.add_argument(
    "--ratio",
    type=float,
    default=1,
    help="proportion of train set used for dataset sampling",
)
parser.add_argument(
    "--test-ratio",
    type=int,
    default=1.0,
    help="proportion of train set used for dataset sampling",
)
parser.add_argument(
    "--dataset", type=str, default="imdb", help="dataset to use for measuring coverage"
)
parser.add_argument(
    "--save-dir",
    type=str,
    default="./coverage/",
    help="dataset to use for measuring coverage",
)
parser.add_argument(
    "--model",
    type=str,
    default="bert-base-uncased",
    help="model f whose weights to use",
)
parser.add_argument("--coverage", type=str, default="attention", help="coverage type")
parser.add_argument("--attack", type=str, default="none", help="attack type")
parser.add_argument(
    "--split", type=str, default="test", help="split to use for measuring coverage"
)
parser.add_argument("--base-only", action="store_true", help="loading only base model")
parser.add_argument("--prespecify-limits", action="store_true", help="prespecify")
args = parser.parse_args()
random_seed(args.seed)


if not args.base_only:
    if args.dataset == "sst2":
        test_model = "textattack/" + str(args.model) + "-" + "SST-2"
    elif args.dataset == "rotten-tomatoes":
        test_model = "textattack/" + str(args.model) + "-" + "rotten_tomatoes"
    else:
        test_model = "textattack/" + str(args.model) + "-" + str(args.dataset)
else:
    test_model = args.model
text_key = "text"
# test_model="textattack/bert-base-uncased-ag-news",
if args.dataset == "sst2":
    text_key = "sentence"
    trainset = HuggingFaceDataset("glue", "sst2", "train", shuffle=True)
    testset = HuggingFaceDataset("glue", "sst2", args.split, shuffle=True)
elif args.dataset == "rotten-tomatoes":
    trainset = HuggingFaceDataset("rotten_tomatoes", None, "train", shuffle=True)
    testset = HuggingFaceDataset("rotten_tomatoes", None, args.split, shuffle=True)
else:
    trainset = HuggingFaceDataset(args.dataset, None, "train", shuffle=True)
    testset = HuggingFaceDataset(args.dataset, None, args.split, shuffle=True)


if args.ratio <= 1.0:
    trainset = trainset[0 : floor(args.ratio * len(trainset))]
else:
    trainset = trainset[0 : floor(args.ratio)]


trainset_str = []
for example in trainset:
    trainset_str.append(example[0][text_key])

if args.test_ratio <= 1.0:
    testset = testset[0 : floor(args.test_ratio * len(testset))]
else:
    testset = testset[0 : floor(args.test_ratio)]

testset_str = []
for example in testset:
    testset_str.append(example[0][text_key])


args.save_dir += "COVER_" + args.coverage + "/"
os.makedirs(args.save_dir, exist_ok=True)
args.save_dir += "SEED_" + str(args.seed) + "_BINS_" + str(args.bins) + "/"
os.makedirs(args.save_dir, exist_ok=True)
args.save_dir += (
    "data_"
    + str(args.dataset)
    + "_model_"
    + str(args.model)
    + "_ratio_"
    + str(args.ratio)
    + "_test_ratio_"
    + str(args.test_ratio)
    + "_L_"
    + str(args.length)
    + "_B_"
    + str(args.base_only)
    + "/"
)


os.makedirs(args.save_dir, exist_ok=True)
args.save_dir += "Attack_" + args.attack + "_limits_" + str(args.prespecify_limits)
os.makedirs(args.save_dir, exist_ok=True)

# make coverage object
coverage = neuronMultiSectionCoverage(
    test_model=test_model,
    max_seq_len=args.length,
    k_m=args.bins,
    coverage=(args.coverage),
    pre_limits=(not (args.coverage == "word") and args.prespecify_limits),
)
print("initializing from training data")
coverage.initialize_from_training_dataset(trainset_str)

print("--" * 50)
print("generating test set!")
print("--" * 50)
num_successes = 0.0
total = 1.0
if args.attack != "none":
    original_model = AutoModelForSequenceClassification.from_pretrained(test_model)
    original_tokenizer = AutoTokenizer(test_model)
    model = HuggingFaceModelWrapper(original_model, original_tokenizer)
    if args.attack == "textfooler":
        attack = TextFoolerJin2019.build(model)
    elif args.attack == "alzantot":
        attack = FasterGeneticAlgorithmJia2019.build(model)
    elif args.attack == "bae":
        attack = BAEGarg2019.build(model)
    elif args.attack == "deepwordbug":
        attack = DeepWordBugGao2018.build(model)
    elif args.attack == "hotflip":
        attack = HotFlipEbrahimi2017.build(model)
    else:
        print("This Attack has not been added!")
        raise NotImplementedError
    results_iterable = attack.attack_dataset(testset, indices=None)
    # save the results too
    results_iterable = [result for result in results_iterable]
    total = len(results_iterable)
    pickle.dump(
        results_iterable, open(os.path.join(args.save_dir, "attack_results"), "wb")
    )
    for n, result in enumerate(results_iterable):
        print("---original: \n", result.original_text())
        print("---perturbed: \n", result.perturbed_text())
        testset_str.append(result.perturbed_text())
        if isinstance(result, SuccessfulAttackResult):
            num_successes += 1


print("=+" * 20)
print("successes: ", num_successes, "total: ", total)
print("rate: ", num_successes / total)
print("--" * 50)
print("length of generated test set: ", len(testset_str))
print("--" * 50)


word_coverage = coverage(testset_str)


print("the coverage: ", word_coverage)

results_file = open(os.path.join(args.save_dir, "stats.txt"), "w")
results_file.write(
    "dataset, model, ratio, length, attack, limits, coverage, num_examples, num_test_examples, seed, split, coverage, num_successes, total\n"
)
results_file.write(
    ",".join(
        [
            args.dataset,
            test_model,
            str(args.ratio),
            str(args.test_ratio),
            str(args.length),
            args.attack,
            str(args.prespecify_limits),
            str(args.coverage),
            str(len(trainset_str)),
            str(len(testset_str)),
            str(args.seed),
            args.split,
            str(word_coverage),
            str(num_successes),
            str(total) + "\n",
        ]
    )
)
results_file.close()
