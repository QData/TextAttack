import os
import time

from test_models import color_text

def log_sep():
    print('\n' + ('-' * 60) + '\n')

def print_gray(s):
    print(color_text(s, 'light_gray'))
    
def main():
    # Change to TextAttack root directory.
    this_file_path = os.path.abspath(__file__)
    test_directory_name = os.path.dirname(this_file_path)
    textattack_root_directory_name = os.path.dirname(test_directory_name)
    os.chdir(textattack_root_directory_name)
    print_gray(f'Executing tests from {textattack_root_directory_name}.')

    # Execute tests.
    start_time = time.time()
    passed_tests = 0
    
    from tests import tests
    for test in tests: 
        log_sep()
        test_passed = test()
        if test_passed:
            passed_tests += 1
    log_sep()
    end_time = time.time()
    print_gray(f'Passed {passed_tests}/{len(tests)} in {end_time-start_time}s.')
    


if __name__ == '__main__':
    # @TODO add argparser and test sizes.
    main()