from helpers import run_command_and_get_result
import pytest

list_test_params = [
    (
        "list_augmentation_recipes",
        "textattack list augmentation-recipes",
        "tests/sample_outputs/list_augmentation_recipes.txt",
    )
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
