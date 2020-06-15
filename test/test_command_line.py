import re
import shlex
import subprocess

import pytest

"""
Attack command-line tests in the format (name, args, sample_output_file)
"""
attack_test_params = [
    #
    # test loading an attack from file
    #
    (
        "attack_from_file",
        (
            "python -m textattack --model cnn-imdb "
            "--attack-from-file test/sample_inputs/attack_from_file.py:Attack "
            "--num-examples 2 --attack-n"
        ),
        "test/sample_outputs/run_attack_from_file.txt",
    ),
    #
    # test interactive mode
    #
    (
        "interactive_mode",
        (
            'printf "All that glitters is not gold\nq\n"',
            "python -m textattack --recipe textfooler --model bert-base-uncased-imdb --interactive",
        ),
        "test/sample_outputs/interactive_mode.txt",
    ),
    #
    # test loading an attack from the transformers model hub
    #
    (
        "attack_from_transformers",
        (
            "python -m textattack --model-from-huggingface "
            "distilbert-base-uncased-finetuned-sst-2-english "
            "--dataset-from-nlp glue:sst2:train --recipe deepwordbug --num-examples 5"
        ),
        "test/sample_outputs/run_attack_transformers_nlp.txt",
    ),
    #
    # test running an attack by loading a model and dataset from file
    #
    (
        "load_model_and_dataset_from_file",
        (
            "python -m textattack --model-from-file test/sample_inputs/sst_model_and_dataset.py "
            "--dataset-from-file test/sample_inputs/sst_model_and_dataset.py "
            "--recipe deepwordbug --num-examples 5"
        ),
        "test/sample_outputs/run_attack_transformers_nlp.txt",
    ),
    #
    # test hotflip on 10 samples from LSTM MR
    #
    (
        "run_attack_hotflip_lstm_mr_4",
        (
            "python -m textattack --model lstm-mr --recipe hotflip "
            "--num-examples 4 --num-examples-offset 13"
        ),
        "test/sample_outputs/run_attack_hotflip_lstm_mr_4.txt",
    ),
    #
    # test deepwordbug on 10 samples from BERT SNLI
    #       (takes about 51s)
    #
    (
        "run_attack_deepwordbug_bert_snli_10",
        (
            "python -m textattack --model bert-base-uncased-snli --recipe deepwordbug --num-examples 10"
        ),
        "test/sample_outputs/run_attack_deepwordbug_bert_snli_10.txt",
    ),
    #
    # test: run_attack deepwordbug attack on 10 samples from LSTM MR
    #                   (takes about 41s)
    #
    (
        "run_attack_deepwordbug_lstm_mr_10",
        ("python -m textattack --model lstm-mr --recipe deepwordbug --num-examples 10"),
        "test/sample_outputs/run_attack_deepwordbug_lstm_mr_10.txt",
    ),
    #
    # test: run_attack targeted classification of class 2 on BERT MNLI with enable_csv
    #   and attack_n set, using the WordNet transformation and beam search with
    #   beam width 2, using language tool constraint, on 10 samples
    #                   (takes about 72s)
    #
    (
        "run_attack_targeted_mnli_misc",
        (
            "python -m textattack --attack-n --goal-function targeted-classification:target_class=2 "
            "--enable-csv --model bert-base-uncased-mnli --num-examples 4 --transformation word-swap-wordnet "
            "--constraints lang-tool repeat stopword --search beam-search:beam_width=2"
        ),
        "test/sample_outputs/run_attack_targetedclassification2_wordnet_langtool_enable_csv_beamsearch2_attack_n_4.txt",
    ),
    #
    # test: run_attack non-overlapping output of class 2 on T5 en->de translation with
    #   attack_n set, using the WordSwapRandomCharacterSubstitution transformation
    #   and greedy word swap, using edit distance constraint, on 6 samples
    #                   (takes about 100s)
    #
    (
        "run_attack_nonoverlapping_t5en2de_randomcharsub_editdistance_wordsperturbed_greedyword",
        (
            "python -m textattack --attack-n --goal-function non-overlapping-output "
            "--model t5-en2de --num-examples 6 --transformation word-swap-random-char-substitution "
            "--constraints edit-distance:12 max-words-perturbed:max_percent=0.75 repeat stopword "
            "--search greedy"
        ),
        "test/sample_outputs/run_attack_nonoverlapping_t5ende_editdistance_bleu.txt",
    ),
    #
    #
    #
]


@pytest.mark.parametrize("name, command, sample_output_file", attack_test_params)
def test_command_line_attack(capsys, name, command, sample_output_file):
    """ Runs attack tests and compares their outputs to a reference file.
    """
    # read in file and create regex
    desired_output = open(sample_output_file, "r").read()
    # regex in sample file look like /.*/
    desired_re = re.escape(desired_output).replace("/\\.\\*/", ".*")
    print("desired_output", desired_output)
    # run command
    if isinstance(command, tuple):
        # Support pipes via tuple of commands
        procs = []
        for i in range(len(command) - 1):
            if i == 0:
                proc = subprocess.Popen(shlex.split(command[i]), stdout=subprocess.PIPE)
            else:
                proc = subprocess.Popen(
                    shlex.split(command[i]),
                    stdout=subprocess.PIPE,
                    stdin=procs[-1].stdout,
                )
            procs.append(proc)
        # Run last commmand
        result = subprocess.run(
            shlex.split(command[-1]), stdin=procs[-1].stdout, capture_output=True
        )
        # Wait for all intermittent processes
        for proc in procs:
            proc.wait()
    else:
        result = subprocess.run(shlex.split(command), capture_output=True)
    # get output and check match
    print("result", result.stdout)
    print("stderr:", result.stderr)
    assert result.stdout is not None
    stdout = result.stdout.decode()
    assert result.stderr is not None
    _ = result.stderr.decode()

    assert re.match(desired_re, stdout)
