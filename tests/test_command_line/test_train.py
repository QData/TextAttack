import os
import re

from helpers import run_command_and_get_result


def test_train_tiny():
    command = "textattack train --model lstm --dataset glue^cola --max-length 32 --batch-size 64 --epochs 1 --pct-dataset .01"

    # Run command and validate outputs.
    result = run_command_and_get_result(command)

    assert result.stdout is not None
    assert result.stderr is not None
    assert result.returncode == 0

    stdout = result.stdout.decode().strip()
    print("stdout =>", stdout)
    stderr = result.stderr.decode().strip()
    print("stderr =>", stderr)

    train_args_json_path = re.findall(
        r"Wrote original training args to (\S+)\.", stderr
    )
    assert len(train_args_json_path) and os.path.exists(train_args_json_path[0])

    train_acc = re.findall(r"Train accuracy: (\S+)", stderr)
    assert train_acc
    train_acc = float(train_acc[0][:-1])  # [:-1] removes percent sign
    assert train_acc > 50

    eval_acc = re.findall(r"Eval accuracy: (\S+)", stderr)
    assert eval_acc
    eval_acc = float(eval_acc[0][:-1])  # [:-1] removes percent sign
    assert eval_acc > 50
