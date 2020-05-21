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
# test: run_attack --interactive
#
register_test(('printf "All that glitters is not gold\nq\n"', 
    'python -m textattack --recipe textfooler --model bert-imdb --interactive'), 
    name='interactive_mode', 
    output_file='local_tests/sample_outputs/interactive_mode.txt', 
    desc='Runs textfooler attack on BERT trained on IMDB using interactive mode')

#
# test: run_attack_parallel textfooler attack on 10 samples from BERT MR
#                   (takes about 81s)
#
register_test('python -m textattack --model bert-mr --recipe textfooler --num-examples 10', 
    name='run_attack_textfooler_bert_mr_10', 
    output_file='local_tests/sample_outputs/run_attack_textfooler_bert_mr_10.txt', 
    desc='Runs attack using TextFooler recipe on BERT using 10 examples from the MR dataset') 

#
# test: run_attack_parallel textfooler attack on 10 samples from BERT SNLI
#                   (takes about 51s)
#
register_test('python -m textattack --model bert-snli --recipe deepwordbug --num-examples 10', 
    name='run_attack_deepwordbug_bert_snli_10', 
    output_file='local_tests/sample_outputs/run_attack_deepwordbug_bert_snli_10.txt', 
    desc='Runs attack using DeepWordBug recipe on BERT using 10 examples from the SNLI dataset')
    
#
# test: run_attack deepwordbug attack on 10 samples from LSTM MR
#                   (takes about 41s)
#
register_test('python -m textattack --model lstm-mr --recipe deepwordbug --num-examples 10', 
    name='run_attack_deepwordbug_lstm_mr_10', 
    output_file='local_tests/sample_outputs/run_attack_deepwordbug_lstm_mr_10.txt', 
    desc='Runs attack using DeepWordBug recipe on LSTM using 10 examples from the MR dataset')
    
#
# test: run_attack targeted classification of class 2 on BERT MNLI with enable_csv
#   and attack_n set, using the WordNet transformation and beam search with 
#   beam width 2, using language tool constraint, on 10 samples
#                   (takes about 72s)
#
register_test(('python -m textattack --attack-n --goal-function targeted-classification:target_class=2 '
    '--enable-csv --model bert-mnli --num-examples 4 --transformation word-swap-wordnet '
    '--constraints lang-tool repeat stopword --search beam-search:beam_width=2'), 
    name='run_attack_targeted2_bertmnli_wordnet_beamwidth_2_enablecsv_attackn', 
    output_file='local_tests/sample_outputs/run_attack_targetedclassification2_wordnet_langtool_enable_csv_beamsearch2_attack_n_4.txt', 
    desc=('Runs attack using targeted classification on class 2 on BERT MNLI with'
        'enable_csv and attack_n set, using the WordNet transformation and beam '
        'search with  beam width 2, using language tool constraint, on 10 samples')
        )
    
#
# test: run_attack non-overlapping output of class 2 on T5 en->de translation with
#   attack_n set, using the WordSwapRandomCharacterSubstitution transformation 
#   and greedy word swap, using edit distance constraint, on 6 samples
#                   (takes about 100s)
#
register_test(('python -m textattack --attack-n --goal-function non-overlapping-output '
    '--model t5-en2de --num-examples 6 --transformation word-swap-random-char-substitution '
    '--constraints edit-distance:12 max-words-perturbed:max_percent=0.75 repeat stopword '
    '--search greedy'), 
    name='run_attack_nonoverlapping_t5en2de_randomcharsub_editdistance_wordsperturbed_greedyword', 
    output_file='local_tests/sample_outputs/run_attack_nonoverlapping_t5ende_editdistance_bleu.txt', 
    desc=('Runs attack using targeted classification on class 2 on BERT MNLI with'
        'enable_csv and attack_n set, using the WordNet transformation and beam '
        'search with  beam width 2, using language tool constraint, on 10 samples')
        )
