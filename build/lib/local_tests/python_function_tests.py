import os
from test_models import PythonFunctionTest

tests = []

def register_test(function, name=None, output_file=None, desc=None):
    if not os.path.exists(output_file):
        raise FileNotFoundError(f'Error creating test {name}: cannot find file {output_file}.')
    output = open(output_file).read()
    tests.append(PythonFunctionTest(
            function, name=name, output=output, desc=desc
        ))


#######################################
##            BEGIN TESTS            ##
#######################################

#
#
#
def check_gpu_count():
    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print(f'Error: detected 0 GPUs. Must run local tests with multiple GPUs. Perhaps you need to configure CUDA?')
    
register_test(check_gpu_count, name='check CUDA', 
    output_file='local_tests/outputs/empty_file.txt', 
    desc='Makes sure CUDA is enabled, properly configured, and detects at least 1 GPU')

#
# test: import textattack
#
def import_textattack():
    import textattack
    
register_test(import_textattack, name='import textattack', 
    output_file='local_tests/outputs/empty_file.txt', 
    desc='Makes sure the textattack module can be imported')