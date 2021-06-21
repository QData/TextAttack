from contextlib import redirect_stderr, redirect_stdout
import io
import pdb
import re

import pytest
import transformers

import textattack

DEBUG = False


def compare_attack_results(stdout, stderr, sample_output_file):
    """Compare outputs from stdout with the expected output."""
    desired_output = open(sample_output_file, "r").read().strip()
    print("desired_output.encoded =>", desired_output.encode())
    print("desired_output =>", desired_output)
    # regex in sample file look like /.*/
    # / is escaped in python 3.6, but not 3.7+, so we support both
    desired_re = (
        re.escape(desired_output)
        .replace("/\\.\\/", ".")
        .replace("/\\.\\*/", ".*")
        .replace("\\/\\.\\*\\/", ".*")
    )

    # get output and check match
    assert stdout is not None
    stdout = stdout.strip()
    print("stdout =>", stdout)
    assert stderr is not None
    stderr = stderr.strip()
    print("stderr =>", stderr)

    if DEBUG and not re.match(desired_re, stdout, flags=re.S):
        pdb.set_trace()
    assert re.match(desired_re, stdout, flags=re.S)


@pytest.mark.slow
def test_kuleshove_nn():
    """API test equivalent to:

    textattack attack --recipe kuleshov --num-examples 2 --model cnn-
    sst2 --attack-n --query-budget 200
    """
    # Catch stdout and stderr
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout):
        with redirect_stderr(stderr):
            colored_model_name = textattack.shared.utils.color_text(
                "cnn-sst2", color="blue", method="ansi"
            )
            textattack.shared.logger.info(
                f"Loading pre-trained TextAttack CNN: {colored_model_name}"
            )
            model = textattack.models.helpers.WordCNNForClassification.from_pretrained(
                "cnn-sst2"
            )
            model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(
                model, model.tokenizer
            )
            dataset = textattack.datasets.HuggingFaceDataset(
                "glue", "sst2", "validation"
            )

            attack = textattack.attack_recipes.Kuleshov2017.build(model_wrapper)
            attack_args = textattack.AttackArgs(
                num_examples=2, query_budget=200, attack_n=True
            )

            attacker = textattack.Attacker(attack, dataset, attack_args)
            attacker.attack_dataset()

    stdout = stdout.getvalue()
    stderr = stderr.getvalue()
    compare_attack_results(
        stdout, stderr, "tests/sample_outputs/kuleshov_cnn_sst_2.txt"
    )


@pytest.mark.slow
def test_gradient_wir():
    """API test equivalent to: textattack attack --model bert-base-uncased-mr.

    --num-examples 3 --num-examples-offset 45 \

    --search greedy-word-wir^wir_method=\'gradient\' --transformation word-swap-embedding --constraints repeat stopword
    """
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout):
        with redirect_stderr(stderr):
            colored_model_name = textattack.shared.utils.color_text(
                "bert-base-uncased-mr", color="blue", method="ansi"
            )
            textattack.shared.logger.info(
                f"Loading pre-trained model from HuggingFace model repository: {colored_model_name}"
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                "textattack/bert-base-uncased-rotten-tomatoes"
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                "textattack/bert-base-uncased-rotten-tomatoes"
            )
            model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
                model, tokenizer
            )
            dataset = textattack.datasets.HuggingFaceDataset(
                "rotten_tomatoes", None, "test"
            )

            search_method = textattack.search_methods.GreedyWordSwapWIR(
                wir_method="gradient"
            )
            transformation = textattack.transformations.WordSwapEmbedding()
            constraints = [
                textattack.constraints.pre_transformation.RepeatModification(),
                textattack.constraints.pre_transformation.StopwordModification(),
            ]
            goal_function = textattack.goal_functions.UntargetedClassification(
                model_wrapper
            )

            attack = textattack.Attack(
                goal_function, constraints, transformation, search_method
            )
            attack_args = textattack.AttackArgs(num_examples=3, num_examples_offset=45)

            attacker = textattack.Attacker(attack, dataset, attack_args)
            attacker.attack_dataset()

    stdout = stdout.getvalue()
    stderr = stderr.getvalue()
    compare_attack_results(
        stdout, stderr, "tests/sample_outputs/run_attack_gradient_greedy_word_wir.txt"
    )


@pytest.mark.slow
def test_flair_pos_tagger():
    """API test equivalent to: textattack attack --model bert-base-uncased-mr.

    --search greedy-word-wir --transformation word-swap-embedding \

    --constraints repeat stopword bert-score^min_bert_score=0.8 part-of-speech^tagger_type=\'flair\' \
    --num-examples 4 --num-examples-offset 10`
    """
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout):
        with redirect_stderr(stderr):
            colored_model_name = textattack.shared.utils.color_text(
                "bert-base-uncased-mr", color="blue", method="ansi"
            )
            textattack.shared.logger.info(
                f"Loading pre-trained model from HuggingFace model repository: {colored_model_name}"
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                "textattack/bert-base-uncased-rotten-tomatoes"
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                "textattack/bert-base-uncased-rotten-tomatoes"
            )
            model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
                model, tokenizer
            )
            dataset = textattack.datasets.HuggingFaceDataset(
                "rotten_tomatoes", None, "test"
            )

            search_method = textattack.search_methods.GreedyWordSwapWIR()
            transformation = textattack.transformations.WordSwapEmbedding()
            constraints = [
                textattack.constraints.pre_transformation.RepeatModification(),
                textattack.constraints.pre_transformation.StopwordModification(),
                textattack.constraints.semantics.BERTScore(min_bert_score=0.8),
                textattack.constraints.grammaticality.PartOfSpeech(tagger_type="flair"),
            ]
            goal_function = textattack.goal_functions.UntargetedClassification(
                model_wrapper
            )

            attack = textattack.Attack(
                goal_function, constraints, transformation, search_method
            )
            attack_args = textattack.AttackArgs(num_examples=4, num_examples_offset=10)

            attacker = textattack.Attacker(attack, dataset, attack_args)
            attacker.attack_dataset()

    stdout = stdout.getvalue()
    stderr = stderr.getvalue()
    compare_attack_results(
        stdout,
        stderr,
        "tests/sample_outputs/run_attack_flair_pos_tagger_bert_score.txt",
    )
