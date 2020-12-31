import textattack
import functools

if __name__ == "__main__":

    def _attack_build_fn(model_name):
        import textattack
        import transformers

        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
            model, tokenizer
        )

        attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)
        return attack

    model_name = "textattack/bert-base-uncased-imdb"
    dataset = textattack.datasets.HuggingFaceDataset("imdb", None, split="test")
    attack_args = textattack.AttackArgs(num_examples=50, parallel=True)
    attack_build_fn = functools.partial(_attack_build_fn, model_name)
    attacker = textattack.Attacker(attack_build_fn, dataset, attack_args)
    attacker.attack_dataset()
