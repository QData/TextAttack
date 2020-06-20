import pytest

from helpers import run_command_and_get_result

augment_test_params = [
    (
        "simple_augment_test",
        "textattack augment --csv tests/sample_inputs/augment.csv.txt --input-column text --outfile augment_test.csv --overwrite",
        "augment_test.csv",
        "tests/sample_outputs/augment_test.csv.txt",
    )
]


@pytest.mark.parametrize(
    "name, command, outfile, sample_output_file", augment_test_params
)
@pytest.mark.slow
def test_command_line_augmentation(name, command, outfile, sample_output_file):
    import os
    
    desired_text = open(sample_output_file).read().strip()

    # Run command and validate outputs.
    result = run_command_and_get_result(command)

    assert result.stdout is not None
    stdout = result.stdout.decode().strip()
    assert stdout == ""

    assert result.stderr is not None
    stderr = result.stderr.decode().strip()
    assert "Wrote 9 augmentations to augment_test.csv" in stderr

    # Ensure CSV file is correct, then delete it.
    assert os.path.exists(outfile)
    outfile_text = open(outfile).read()
    assert outfile_text == desired_text
    os.remove(outfile)
