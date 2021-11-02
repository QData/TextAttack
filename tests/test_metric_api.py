def test_perplexity():
    from textattack.attack_results import SuccessfulAttackResult
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
                AttackedText(sample_atck_text), None, None, None, None, None, None
            ),
        )
    ]
    ppl = Perplexity(model_name="distilbert-base-uncased").calculate(results)

    assert int(ppl["avg_original_perplexity"]) == int(81.95)


def test_use():
    import transformers

    from textattack import AttackArgs, Attacker
    from textattack.attack_recipes import DeepWordBugGao2018
    from textattack.datasets import HuggingFaceDataset
    from textattack.metrics.quality_metrics import USEMetric
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

    usem = USEMetric().calculate(results)

    assert usem["avg_attack_use_score"] == 0.76
