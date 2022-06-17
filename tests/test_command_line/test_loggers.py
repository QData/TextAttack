import json
import os

from helpers import run_command_and_get_result
import pytest

DEBUG = False

"""
Attack command-line tests in the format (name, args, sample_output_file)
"""

"""
 list_test_params data structure requires
 1) test name
 2) logger filetype - json/text/csv. # Future Work : Tests for Wandb and Visdom
 3) logger file name
 4) sample log file
"""

list_test_params = [
    (
        "json_summary_logger",
        "json",
        "textattack attack --recipe deepwordbug --model lstm-mr --num-examples 2 --log-summary-to-json attack_summary.json",
        "attack_summary.json",
        "tests/sample_outputs/json_attack_summary.json",
    ),
    (
        "csv_logger",
        "csv",
        "textattack attack --recipe deepwordbug --model lstm-mr --num-examples 2 --log-to-csv attack_log.csv",
        "attack_log.csv",
        "tests/sample_outputs/csv_attack_log.csv",
    ),
    (
        "txt_logger",
        "txt",
        "textattack attack --recipe deepwordbug --model lstm-mr --num-examples 2 --log-to-txt attack_log.txt",
        "attack_log.txt",
        "tests/sample_outputs/txt_attack_log.txt",
    ),
]


@pytest.mark.parametrize(
    "name, filetype, command, test_log_file, sample_log_file", list_test_params
)
def test_logger(name, filetype, command, test_log_file, sample_log_file):
    # Run command and validate outputs.
    result = run_command_and_get_result(command)

    assert result.stdout is not None
    assert result.stderr is not None
    assert result.returncode == 0
    assert os.path.exists(test_log_file), f"{test_log_file} did not get generated"

    if filetype == "json":
        with open(sample_log_file) as f:
            desired_dictionary = json.load(f)

        with open(test_log_file) as f:
            test_dictionary = json.load(f)

        assert (
            desired_dictionary == test_dictionary
        ), f"{filetype} file {test_log_file} differs from {sample_log_file}"

    elif filetype == "csv" or filetype == "txt":
        assert (
            os.system(f"diff {test_log_file} {sample_log_file}") == 0
        ), f"{filetype} file {test_log_file} differs from {sample_log_file}"

    # cleanup
    os.remove(test_log_file)


#     result = run_command_and_get_result(command)
#     stdout = result.stdout.decode().strip()
#     print("stdout =>", stdout)
#     stderr = result.stderr.decode().strip()
#     print("stderr =>", stderr)

#     assert stdout == desired_text
