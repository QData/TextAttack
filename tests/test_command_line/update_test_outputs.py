"""
Re-runs command-line tests and places their outputs in the sample files. 

This is useful for large changes, but be wary: the outputs still may need to be
manually edited to account for variance between runs.
"""
from helpers import run_command_and_get_result
from test_attack import attack_test_params
from test_augment import augment_test_params
from test_list import list_test_params


def update_test(command, outfile):
    if isinstance(command, str):
        command = (command,)
    command = command + (f"tee {outfile}",)
    print("\n".join(f"> {c}" for c in command))
    run_command_and_get_result(command)


def main():
    #### `textattack attack` tests ####
    for _, command, outfile in attack_test_params:
        update_test(command, outfile)
    #### `textattack augment` tests ####
    for _, command, outfile, __ in augment_test_params:
        update_test(command, outfile)
    #### `textattack list` tests
    for _, command, outfile in list_test_params:
        update_test(command, outfile)


if __name__ == "__main__":
    main()
