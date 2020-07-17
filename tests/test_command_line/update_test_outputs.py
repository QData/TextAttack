"""Re-runs command-line tests and places their outputs in the sample files.

This is useful for large changes, but be wary: the outputs still may
need to be manually edited to account for variance between runs.
"""
from helpers import run_command_and_get_result
from test_attack import attack_test_params
from test_augment import augment_test_params
from test_list import list_test_params


def update_test(command, outfile, add_magic_str=False):
    if isinstance(command, str):
        print(">", command)
    else:
        print("\n".join(f"> {c}" for c in command))
    result = run_command_and_get_result(command)
    stdout = result.stdout.decode().strip()
    if add_magic_str:
        # add magic string to beginning
        magic_str = "/.*/"
        stdout = magic_str + stdout
        # add magic string after attack
        mid_attack_str = "\n--------------------------------------------- Result 1"
        stdout.replace(mid_attack_str, magic_str + mid_attack_str)
    # write to file
    open(outfile, "w").write(stdout + "\n")


def main():
    #### `textattack attack` tests ####
    for _, command, outfile in attack_test_params:
        update_test(command, outfile, add_magic_str=True)
    #### `textattack augment` tests ####
    for _, command, outfile, __ in augment_test_params:
        update_test(command, outfile)
    #### `textattack list` tests
    for _, command, outfile in list_test_params:
        update_test(command, outfile)


if __name__ == "__main__":
    main()
