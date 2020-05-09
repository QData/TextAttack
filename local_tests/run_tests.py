import argparse
import os
import time

from test_lists import tests
from test_models import color_text

def log_sep():
    print('\n' + ('-' * 60) + '\n')

def print_gray(s):
    print(color_text(s, 'light_gray'))

def change_to_root_dir():
    # Change to TextAttack root directory.
    this_file_path = os.path.abspath(__file__)
    test_directory_name = os.path.dirname(this_file_path)
    textattack_root_directory_name = os.path.dirname(test_directory_name)
    os.chdir(textattack_root_directory_name)
    print_gray(f'Executing tests from {textattack_root_directory_name}.')
    
def run_all_tests():
    change_to_root_dir()
    start_time = time.time()
    passed_tests = 0
    
    for test in tests: 
        log_sep()
        test_passed = test()
        if test_passed:
            passed_tests += 1
    log_sep()
    end_time = time.time()
    print_gray(f'Passed {passed_tests}/{len(tests)} in {end_time-start_time}s.')
    

def run_tests_by_name(test_names):
    test_names = set(test_names)
    start_time = time.time()
    passed_tests = 0
    executed_tests = 0
    for test in tests: 
        if test.name not in test_names:
            continue
        log_sep()
        test_passed = test()
        if test_passed:
            passed_tests += 1
        executed_tests += 1
        test_names.remove(test.name)
    log_sep()
    end_time = time.time()
    print_gray(f'Passed {passed_tests}/{executed_tests} in {end_time-start_time}s.')
    
    if len(test_names):
        print(f'Tests not executed: {",".join(test_names)}')

def parse_args():
    all_test_names = [t.name for t in tests]
    parser = argparse.ArgumentParser(description='Run TextAttack local tests.')
    parser.add_argument('--tests', default=None, nargs='+', choices=all_test_names,
                    help='names of specific tests to run')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.tests:
        run_tests_by_name(args.tests)
    else:
        run_all_tests()