def test_perplexity():
    import transformers

    import textattack
    from textattack.attack_results import SuccessfulAttackResult
    from textattack.goal_function_results.classification_goal_function_result import (
        ClassificationGoalFunctionResult,
    )
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

    ppl = textattack.metrics.quality_metrics.Perplexity().calculate(results)

    assert int(ppl["avg_original_perplexity"]) == int(1854.74)
    assert int(ppl["avg_attack_perplexity"]) == int(4214.01)
