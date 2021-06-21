from contextlib import redirect_stderr, redirect_stdout
import io
import logging
import os
import re

import textattack


def test_adv_train():
    stdout = io.StringIO()
    stderr = io.StringIO()
    logger_output = io.StringIO()

    textattack.shared.utils.logger.addHandler(logging.StreamHandler(logger_output))

    with redirect_stdout(stdout):
        with redirect_stderr(stderr):
            model = textattack.models.helpers.WordCNNForClassification()
            model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(
                model, model.tokenizer
            )

            train_dataset = textattack.datasets.HuggingFaceDataset(
                "rotten_tomatoes", None, "train"
            )
            eval_dataset = textattack.datasets.HuggingFaceDataset(
                "rotten_tomatoes", None, "test"
            )
            # TODO Switch to faster word swap attack recipe
            attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
            training_args = textattack.TrainingArgs(
                num_epochs=2,
                num_clean_epochs=1,
                num_train_adv_examples=500,
                query_budget_train=200,
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

    stdout = stdout.getvalue().strip()
    stderr = stderr.getvalue().strip()
    logger_output = logger_output.getvalue().strip()

    print("stdout =>", stdout)
    print("stderr =>", stderr)
    print("logger =>", logger_output)
    train_args_json_path = re.findall(
        r"Wrote original training args to (\S+)\.", logger_output
    )
    assert len(train_args_json_path) and os.path.exists(train_args_json_path[0])

    train_acc = re.findall(r"Train accuracy: (\S+)", logger_output)
    assert train_acc
    train_acc = float(train_acc[1][:-1])  # [:-1] removes percent sign
    assert train_acc > 60

    eval_acc = re.findall(r"Eval accuracy: (\S+)", logger_output)
    assert eval_acc
    eval_acc = float(eval_acc[1][:-1])  # [:-1] removes percent sign
    assert eval_acc > 65
