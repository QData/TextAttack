import textattack
import torch
import transformers
from textattack.models.wrappers import HuggingFaceModelWrapper
import pickle
import functools


if __name__ == "__main__":
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-rotten-tomatoes"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "textattack/bert-base-uncased-rotten-tomatoes", use_fast=True
    )
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    # eval_dataset = textattack.datasets.HuggingFaceDataset("rotten_tomatoes", split="test")
    # attack_args = textattack.AttackArgs(num_examples=100, parallel=True)
    # attacker = textattack.Attacker(attack, eval_dataset, attack_args=attack_args)
    # attacker.attack_dataset()

    train_dataset = textattack.datasets.HuggingFaceDataset(
        "rotten_tomatoes", split="train"
    )
    eval_dataset = textattack.datasets.HuggingFaceDataset(
        "rotten_tomatoes", split="test"
    )

    training_args = textattack.TrainingArgs(
        num_epochs=1,
        num_clean_epochs=1,
        parallel=True,
        eval_adversarial_robustness=True,
        num_eval_adv_examples=100,
        num_train_adv_examples=1000,
        query_budget_train=150,
        checkpoint_interval_epochs=1,
        log_to_tb=True,
        load_best_model_at_end=True,
    )
    trainer = textattack.Trainer(
        model_wrapper,
        "classification",
        attack,
        train_dataset,
        eval_dataset,
        training_args,
    )
    trainer.train()
    # trainer.eval()
