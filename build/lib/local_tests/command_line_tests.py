import os
from test_models import CommandLineTest

tests = []

def register_test(command, name=None, output_file=None, desc=None):
    if not os.path.exists(output_file):
        raise FileNotFoundError(f'Error creating test {name}: cannot find file {output_file}.')
    output = open(output_file).read()
    tests.append(CommandLineTest(
            command, name=name, output=output, desc=desc
        ))


#######################################
##            BEGIN TESTS            ##
#######################################

#
# test: run_attack_parallel textfooler attack on 10 samples from BERT MR
#                   (takes about 81s)
#
register_test('python scripts/run_attack.py --model bert-mr --recipe textfooler --num_examples 10', 
    name='run_attack_textfooler_bert_mr_10', 
    output_file='local_tests/outputs/run_attack_textfooler_bert_mr_10.txt', 
    desc='Runs attack using TextFooler recipe on BERT using 10 examples from the MR dataset') 

#
# test: run_attack_parallel textfooler attack on 10 samples from BERT SNLI
#                   (takes about 51s)
#
register_test('python scripts/run_attack.py --model bert-snli --recipe textfooler --num_examples 10', 
    name='run_attack_textfooler_bert_snli_10', 
    output_file='local_tests/outputs/run_attack_textfooler_bert_snli_10.txt', 
    desc='Runs attack using TextFooler recipe on BERT using 10 examples from the SNLI dataset')
    
#
# test: run_attack deepwordbug attack on 10 samples from LSTM MR
#                   (takes about 41s)
#
register_test('python scripts/run_attack.py --model lstm-mr --recipe deepwordbug --num_examples 10', 
    name='run_attack_deepwordbug_lstm_mr_10', 
    output_file='local_tests/outputs/run_attack_deepwordbug_lstm_mr_10.txt', 
    desc='Runs attack using DeepWordBUG recipe on LSTM using 10 examples from the MR dataset')
    
#
# test: run_attack targeted classification of class 2 on BERT MNLI with enable_csv
#   and attack_n set, using the WordNet transformation and beam search with 
#   beam width 2, using language tool constraint, on 10 samples
#                   (takes about 171s)
#
register_test(('python scripts/run_attack.py --attack_n --goal_function targeted-classification:target_class=2 '
    '--enable_csv --model bert-mnli --num_examples 10 --transformation word-swap-wordnet '
    '--constraints lang-tool --attack beam-search:beam_width=2'), 
    name='run_attack_targeted2_bertmnli_wordnet_beamwidth_2_enablecsv_attackn', 
    output_file='local_tests/outputs/run_attack_targetedclassification2_wordnet_langtool_enable_csv_beamsearch2_attack_n_10.txt', 
    desc=('Runs attack using targeted classification on class 2 on BERT MNLI with'
        'enable_csv and attack_n set, using the WordNet transformation and beam '
        'search with  beam width 2, using language tool constraint, on 10 samples')
        )