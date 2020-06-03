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
        raise ValueError(f'detected 0 GPUs. Must run local tests with a GPU. Perhaps you need to configure CUDA?')
    
register_test(check_gpu_count, name='check_cuda', 
    output_file='local_tests/sample_outputs/empty_file.txt', 
    desc='Makes sure CUDA is enabled, properly configured, and detects at least 1 GPU')

#
# test: import textattack
#
def import_textattack():
    import textattack
    
register_test(import_textattack, name='import_textattack', 
    output_file='local_tests/sample_outputs/empty_file.txt', 
    desc='Makes sure the textattack module can be imported')
#
# test: import augmenter
#
def use_embedding_augmenter():
    from textattack.augmentation import EmbeddingAugmenter
    augmenter = EmbeddingAugmenter(transformations_per_example=64)
    s = 'There is nothing either good or bad, but thinking makes it so.'
    augmented_text_list = augmenter.augment(s)
    augmented_s = 'There is nothing either good or unfavourable, but thinking makes it so.'
    assert augmented_s in augmented_text_list
    
    
register_test(use_embedding_augmenter, name='use_embedding_augmenter', 
    output_file='local_tests/sample_outputs/empty_file.txt', 
    desc='Imports EmbeddingAugmenter and augments a single sentence')
