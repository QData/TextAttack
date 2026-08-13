"""Structural and functional tests for LEAP2023 against its closest existing
recipe, PSOZang2020.

Both recipes attack classification models with a population-based Particle
Swarm Optimization search over word substitutions -- LEAP2023 is a
Levy-flight/adaptive-inertia variant of the same PSO family PSOZang2020
already implements in this codebase (see
``textattack/search_methods/particle_swarm_optimization_leap.py`` for the
full lineage and references).

These tests only check structure and that neither recipe errors out; they
don't measure attack quality (success rate, query efficiency, perturbation
size). For that, see ``tests/benchmark_leap_vs_pso.py`` (a manual script,
not run in CI/here, since a meaningful sample takes minutes to run) and the
"Benchmark" section of ``LEAP2023``'s docstring
(``textattack/attack_recipes/leap_2023.py``) for reproducible results.
"""
import pytest

from textattack.attack_recipes import LEAP2023, PSOZang2020
from textattack.constraints.pre_transformation import (
    MaxModificationRate,
    RepeatModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import (
    ParticleSwarmOptimization,
    ParticleSwarmOptimizationLEAP,
)
from textattack.transformations import WordSwapHowNet, WordSwapWordNet


def test_leap_recipe_shares_pso_lineage_with_pso_zang_2020():
    """LEAP2023's search method should be a specialization of the same
    ParticleSwarmOptimization family PSOZang2020 uses, not an unrelated
    implementation."""
    import transformers

    from textattack.models.wrappers import HuggingFaceModelWrapper

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    leap_attack = LEAP2023.build(model_wrapper)
    pso_attack = PSOZang2020.build(model_wrapper)

    # LEAP's search method subclasses the same PSO family PSOZang2020 uses,
    # rather than being an unrelated implementation.
    assert isinstance(leap_attack.search_method, ParticleSwarmOptimizationLEAP)
    assert isinstance(leap_attack.search_method, ParticleSwarmOptimization)
    assert isinstance(pso_attack.search_method, ParticleSwarmOptimization)
    assert not isinstance(pso_attack.search_method, ParticleSwarmOptimizationLEAP)

    # Both recipes use the same goal function and default population budget.
    assert isinstance(leap_attack.goal_function, UntargetedClassification)
    assert isinstance(pso_attack.goal_function, UntargetedClassification)
    assert leap_attack.search_method.pop_size == pso_attack.search_method.pop_size
    assert leap_attack.search_method.max_iters == pso_attack.search_method.max_iters

    # LEAP swaps WordNet synonyms and caps total modification rate;
    # PSOZang2020 swaps HowNet sememes and only prevents re-modifying a word.
    assert isinstance(leap_attack.transformation, WordSwapWordNet)
    assert isinstance(pso_attack.transformation, WordSwapHowNet)
    assert any(
        isinstance(c, MaxModificationRate)
        for c in leap_attack.pre_transformation_constraints
    ), "LEAP2023 should constrain the total fraction of words modified"
    assert any(
        isinstance(c, RepeatModification)
        for c in pso_attack.pre_transformation_constraints
    ), "PSOZang2020 should prevent re-modifying an already-swapped word"


@pytest.mark.slow
def test_leap_and_pso_zang_2020_attack_without_error():
    """Run both recipes end-to-end on the same tiny sample and confirm
    neither raises -- a regression guard for the LEAP search method's
    perform_search wiring (mutation step, omega/velocity bookkeeping)
    against the working PSOZang2020 implementation it was adapted from."""
    import transformers

    from textattack import AttackArgs, Attacker
    from textattack.attack_results import (
        FailedAttackResult,
        SkippedAttackResult,
        SuccessfulAttackResult,
    )
    from textattack.datasets import HuggingFaceDataset
    from textattack.models.wrappers import HuggingFaceModelWrapper

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    dataset = HuggingFaceDataset("glue", "sst2", split="train")

    # PSOZang2020's WordSwapHowNet calls AttackedText.pos_of_word_index, which
    # lazily loads a flair POS-tagger into a module-level cache
    # (textattack.shared.utils.strings._flair_pos_tagger) with no
    # invalidation. Loading it here has been observed to corrupt a
    # *different* flair model (an NER tagger) loaded later in the same
    # process by tests/test_transformations.py's flair-based tests -- reset
    # it afterward so this test doesn't leak that state into later tests.
    import textattack.shared.utils.strings as ta_strings

    try:
        for recipe in (LEAP2023, PSOZang2020):
            attack = recipe.build(model_wrapper)
            attack_args = AttackArgs(
                num_examples=1,
                query_budget=300,
                random_seed=765,
                disable_stdout=True,
            )
            attacker = Attacker(attack, dataset, attack_args)
            results = attacker.attack_dataset()

            assert len(results) == 1
            assert isinstance(
                results[0],
                (SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult),
            )
    finally:
        ta_strings._flair_pos_tagger = None
