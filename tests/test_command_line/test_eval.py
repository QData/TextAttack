from helpers import run_command_and_get_result
import pytest

list_test_params = [
    (
        "eval_model_hub_rt",
        "textattack eval --model-from-huggingface textattack/distilbert-base-uncased-rotten-tomatoes --dataset-from-nlp rotten_tomatoes --num-examples 4",
        "tests/sample_outputs/eval_model_hub_rt.txt",
    ),
    (
        "eval_mnli",
        "textattack eval --model bert-base-uncased-mnli --dataset-from-nlp glue^mnli --num-examples 10",
        "tests/sample_outputs/eval_mnli.txt",
    ),
]


@pytest.mark.parametrize("name, command, sample_output_file", list_test_params)
def test_command_line_list(name, command, sample_output_file):
    desired_text = open(sample_output_file).read().strip()

    # Run command and validate outputs.
    result = run_command_and_get_result(command)

    assert result.stdout is not None
    assert result.stderr is not None

    stdout = result.stdout.decode().strip()
    print("stdout =>", stdout)
    stderr = result.stderr.decode().strip()
    print("stderr =>", stderr)

    assert stdout == desired_text

    assert result.returncode == 0
