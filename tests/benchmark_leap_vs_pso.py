"""Benchmark: LEAP2023 vs. a WordNet-transformation variant of vanilla PSO.

This is a manual benchmark script, not a pytest test -- it is not collected
by CI, since a meaningful sample size takes minutes to run. It exists to
make the comparison used to write the "Benchmark" section of
``LEAP2023``'s docstring (``textattack/attack_recipes/leap_2023.py``)
reproducible, and to make it easy to re-run against other models/datasets/
budgets.

Usage::

    python tests/benchmark_leap_vs_pso.py [--num-examples N] [--query-budget Q]

Methodology
-----------
``PSOZang2020`` normally uses HowNet-based candidates, while ``LEAP2023``
uses WordNet -- comparing them directly conflates "which candidate word
pool is better for this input" with "which search algorithm is better".
This script instead builds a vanilla-PSO attack with WordNet swapped in,
so goal function, transformation, constraints, and search hyperparameters
(pop_size=60, max_iters=20, post_turn_check=True, max_turn_retries=20) are
identical between the two attacks; only the search method's internals
(LEAP's Levy-flight/adaptive-inertia/greedy mutation vs. vanilla PSO's
uniform-velocity/linear-decay/probabilistic mutation) differ.
"""

import argparse
import time

from textattack import AttackArgs, Attacker
from textattack.attack import Attack
from textattack.attack_recipes import LEAP2023
from textattack.attack_results import SuccessfulAttackResult
from textattack.constraints.pre_transformation import (
    MaxModificationRate,
    StopwordModification,
)
from textattack.datasets import HuggingFaceDataset
from textattack.goal_functions import UntargetedClassification
from textattack.models.helpers import WordCNNForClassification
from textattack.models.wrappers import PyTorchModelWrapper
from textattack.search_methods import ParticleSwarmOptimization
from textattack.transformations import WordSwapWordNet

RANDOM_SEED = 765


def build_pso_wordnet(model_wrapper):
    """Vanilla ParticleSwarmOptimization with LEAP2023's transformation,
    constraints, and search hyperparameters, so it differs from LEAP2023 only
    in the search method's internals."""
    transformation = WordSwapWordNet()
    constraints = [MaxModificationRate(max_rate=0.16), StopwordModification()]
    goal_function = UntargetedClassification(model_wrapper)
    search_method = ParticleSwarmOptimization(
        pop_size=60, max_iters=20, post_turn_check=True, max_turn_retries=20
    )
    return Attack(goal_function, constraints, transformation, search_method)


def run(name, attack, dataset, num_examples, query_budget):
    start = time.time()
    attack_args = AttackArgs(
        num_examples=num_examples,
        query_budget=query_budget,
        random_seed=RANDOM_SEED,
        disable_stdout=True,
        silent=True,
    )
    attacker = Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()
    elapsed = time.time() - start

    n = len(results)
    successes = [r for r in results if isinstance(r, SuccessfulAttackResult)]
    n_success = len(successes)
    avg_queries = sum(r.num_queries for r in results) / n if n else float("nan")
    if successes:
        perturbed_pcts = []
        for r in successes:
            orig_words = r.original_result.attacked_text.words
            pert_words = r.perturbed_result.attacked_text.words
            n_diff = sum(1 for a, b in zip(orig_words, pert_words) if a != b)
            perturbed_pcts.append(100.0 * n_diff / len(orig_words))
        avg_perturbed_pct = sum(perturbed_pcts) / len(perturbed_pcts)
    else:
        avg_perturbed_pct = float("nan")

    print(f"\n=== {name} ===")
    print(f"  Examples run:        {n}")
    print(f"  Successful attacks:  {n_success} ({100.0 * n_success / n:.1f}%)")
    print(f"  Avg queries/example: {avg_queries:.1f}")
    print(f"  Avg words perturbed (successes only): {avg_perturbed_pct:.2f}%")
    print(f"  Wall time:           {elapsed:.1f}s")
    return {
        "name": name,
        "n": n,
        "n_success": n_success,
        "avg_queries": avg_queries,
        "avg_perturbed_pct": avg_perturbed_pct,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument(
        "--query-budget",
        type=int,
        default=2000,
        help="Pass 0 for an unrestricted query budget.",
    )
    args = parser.parse_args()
    query_budget = args.query_budget or None

    model = WordCNNForClassification.from_pretrained("cnn-ag-news")
    model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    dataset = HuggingFaceDataset("ag_news", None, split="test")

    leap_attack = LEAP2023.build(model_wrapper)
    pso_wordnet_attack = build_pso_wordnet(model_wrapper)

    leap_stats = run(
        "LEAP2023 (WordNet)", leap_attack, dataset, args.num_examples, query_budget
    )
    pso_stats = run(
        "PSO (WordNet, revised)",
        pso_wordnet_attack,
        dataset,
        args.num_examples,
        query_budget,
    )

    print("\n=== Summary ===")
    print(
        f"{'Method':<25}{'Success%':<12}{'AvgQueries':<14}{'AvgPerturb%':<14}{'Time(s)'}"
    )
    for s in (leap_stats, pso_stats):
        print(
            f"{s['name']:<25}"
            f"{100.0 * s['n_success'] / s['n']:<12.1f}"
            f"{s['avg_queries']:<14.1f}"
            f"{s['avg_perturbed_pct']:<14.2f}"
            f"{s['elapsed']:.1f}"
        )


if __name__ == "__main__":
    main()
