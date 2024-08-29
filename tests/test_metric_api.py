import pytest


def test_perplexity():
    from textattack.attack_results import FailedAttackResult, SuccessfulAttackResult
    from textattack.goal_function_results.classification_goal_function_result import (
        ClassificationGoalFunctionResult,
    )
    from textattack.metrics.quality_metrics import Perplexity
    from textattack.shared.attacked_text import AttackedText

    sample_text = "hide new secretions from the parental units "
    sample_atck_text = "Ehide enw secretions from the parental units "

    results = [
        SuccessfulAttackResult(
            ClassificationGoalFunctionResult(
                AttackedText(sample_text), None, None, None, None, None, None
            ),
            ClassificationGoalFunctionResult(
                AttackedText(
                    sample_atck_text), None, None, None, None, None, None
            ),
        )
    ]
    ppl = Perplexity(model_name="distilbert-base-uncased").calculate(results)

    assert int(ppl["avg_original_perplexity"]) == int(81.95)

    results = [
        FailedAttackResult(
            ClassificationGoalFunctionResult(
                AttackedText(sample_text), None, None, None, None, None, None
            ),
        )
    ]

    Perplexity(model_name="distilbert-base-uncased").calculate(results)

    ppl = Perplexity(model_name="distilbert-base-uncased")
    texts = [sample_text]
    ppl.ppl_tokenizer.encode(" ".join(texts), add_special_tokens=True)

    encoded = ppl.ppl_tokenizer.encode(" ".join([]), add_special_tokens=True)
    assert len(encoded) > 0


def test_perplexity_empty_results():
    from textattack.metrics.quality_metrics import Perplexity

    ppl = Perplexity()
    with pytest.raises(ValueError):
        ppl.calculate([])

    ppl = Perplexity("gpt2")
    with pytest.raises(ValueError):
        ppl.calculate([])

    ppl = Perplexity(model_name="distilbert-base-uncased")
    ppl_values = ppl.calculate([])

    assert "avg_original_perplexity" in ppl_values
    assert "avg_attack_perplexity" in ppl_values


def test_perplexity_no_model():
    from textattack.attack_results import FailedAttackResult, SuccessfulAttackResult
    from textattack.goal_function_results.classification_goal_function_result import (
        ClassificationGoalFunctionResult,
    )
    from textattack.metrics.quality_metrics import Perplexity
    from textattack.shared.attacked_text import AttackedText

    sample_text = "hide new secretions from the parental units "
    sample_atck_text = "Ehide enw secretions from the parental units "

    results = [
        SuccessfulAttackResult(
            ClassificationGoalFunctionResult(
                AttackedText(sample_text), None, None, None, None, None, None
            ),
            ClassificationGoalFunctionResult(
                AttackedText(
                    sample_atck_text), None, None, None, None, None, None
            ),
        )
    ]

    ppl = Perplexity()
    ppl_values = ppl.calculate(results)

    assert "avg_original_perplexity" in ppl_values
    assert "avg_attack_perplexity" in ppl_values


def test_perplexity_calc_ppl():
    from textattack.metrics.quality_metrics import Perplexity

    ppl = Perplexity("gpt2")
    with pytest.raises(ValueError):
        ppl.calc_ppl([])


def test_use():
    import transformers

    from textattack import AttackArgs, Attacker
    from textattack.attack_recipes import DeepWordBugGao2018
    from textattack.datasets import HuggingFaceDataset
    from textattack.metrics.quality_metrics import MeteorMetric
    from textattack.models.wrappers import HuggingFaceModelWrapper

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    attack = DeepWordBugGao2018.build(model_wrapper)
    dataset = HuggingFaceDataset("glue", "sst2", split="train")
    attack_args = AttackArgs(
        num_examples=1,
        log_to_csv="log.csv",
        checkpoint_interval=5,
        checkpoint_dir="checkpoints",
        disable_stdout=True,
    )
    attacker = Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()

    usem = MeteorMetric().calculate(results)

    assert usem["avg_attack_meteor_score"] == 0.71


def test_metric_recipe():
    import transformers

    from textattack import AttackArgs, Attacker
    from textattack.attack_recipes import DeepWordBugGao2018
    from textattack.datasets import HuggingFaceDataset
    from textattack.metrics.recipe import AdvancedAttackMetric
    from textattack.models.wrappers import HuggingFaceModelWrapper

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    attack = DeepWordBugGao2018.build(model_wrapper)
    dataset = HuggingFaceDataset("glue", "sst2", split="train")
    attack_args = AttackArgs(
        num_examples=1,
        log_to_csv="log.csv",
        checkpoint_interval=5,
        checkpoint_dir="checkpoints",
        disable_stdout=True,
    )
    attacker = Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()

    adv_score = AdvancedAttackMetric(
        ["meteor_score", "perplexity"]).calculate(results)
    assert adv_score["avg_attack_meteor_score"] == 0.71


def test_metric_ad_hoc():
    from textattack.metrics.quality_metrics import Perplexity
    from textattack.metrics.recipe import AdvancedAttackMetric

    metrics = AdvancedAttackMetric()
    metrics.add_metric("perplexity", Perplexity(
        model_name="distilbert-base-uncased"))

    metric_results = metrics.calculate([])

    assert "perplexity" in metric_results
