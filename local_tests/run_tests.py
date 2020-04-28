import os
import time
            
def log_sep():
    print('\n' + ('-' * 60) + '\n')
    
def main():
    # Change to TextAttack root directory.
    this_file_path = os.path.abspath(__file__)
    test_directory_name = os.path.dirname(this_file_path)
    textattack_root_directory_name = os.path.dirname(test_directory_name)
    os.chdir(textattack_root_directory_name)
    print(f'Executing tests from {textattack_root_directory_name}.')

    # Execute tests.
    start_time = time.time()
    from tests import tests
    for test in tests: 
        log_sep()
        test()
    log_sep()
    end_time = time.time()
    print(f'Executed {len(tests)} in {end_time-start_time}s.')
    


if __name__ == '__main__':
    # @TODO add argparser and test sizes.
    main()