def test_perplexity():
    from collections import OrderedDict

    import transformers

    import textattack

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
    sample_text = "hide new secretions from the parental units"
    dataset = [(OrderedDict("text", sample_text), 0)]
    attack_args = textattack.AttackArgs(
        num_examples=1,
        log_to_csv="log.csv",
        checkpoint_interval=5,
        checkpoint_dir="checkpoints",
        disable_stdout=False,
        enable_advance_metrics=True,
    )
    attacker = textattack.Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()
    ppl = textattack.metrics.quality_metrics.Perplexity().calculate(results)

    assert int(ppl["avg_original_perplexity"]) == int(1854.74)
