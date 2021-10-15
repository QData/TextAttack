def test_imports():
    from textattack.metrics.quality_metrics import Perplexity
    from textattack.metrics.quality_metrics import USEMetric


def test_perplexity():
    import textattack
    import transformers

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
    dataset = textattack.datasets.HuggingFaceDataset("glue", "sst2", split="train")
    attack_args = textattack.AttackArgs(
        num_examples=1,
        log_to_csv="log.csv",
        checkpoint_interval=5,
        checkpoint_dir="checkpoints",
        disable_stdout=True,
    )
    attacker = textattack.Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()
    ppl = textattack.metrics.quality_metrics.Perplexity().calculate(results)

    assert ppl["avg_original_perplexity"] == 1854.74
