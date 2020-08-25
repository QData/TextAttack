from helpers import run_command_and_get_result
import pytest

eval_test_params = [
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


@pytest.mark.parametrize("name, command, sample_output_file", eval_test_params)
def test_command_line_eval(name, command, sample_output_file):
    """Tests the command-line function, `textattack eval`.

    Different from other tests, this one compares the sample output file
    to *stderr* output of the evaluation.
    """
    desired_text = open(sample_output_file).read().strip()

    # Run command and validate outputs.
    result = run_command_and_get_result(command)

    assert result.stdout is not None
    assert result.stderr is not None

    stdout = result.stdout.decode().strip()
    print("stdout =>", stdout)
    stderr = result.stderr.decode().strip()
    print("stderr =>", stderr)

    assert stderr == desired_text

    assert result.returncode == 0
