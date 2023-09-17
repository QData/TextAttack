


import checklist
import copy
import torch
import random
import numpy as np
import pickle
import wandb


from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
from sst_model import *

from textattack.coverage import neuronMultiSectionCoverage
from textattack.datasets import HuggingFaceDataset
from textattack.metrics import *
from coverage_args import *
from repr_utils import *
from coverage_utils import *
TYPE_MAP = {
            MFT: 'MFT',
            INV: 'INV',
            DIR: 'DIR',
        }

args = get_args()

set_seed(args.seed)
wandb.init()
wandb.config.update(args)
wandb.init(
    project="coverage",
    notes="vanilla coverage only",
    tags=["coverage", "bert"],
    config=wandb.config,
    )
suite_path_dicts = {
    'sentiment' : 'sentiment/sentiment_suite.pkl', 
    'qqp' : 'qqp/qqp_suite.pkl', 
    'mc' : 'squad/squad_suite.pkl'
}
suite_path = './CHECKLIST_DATA/release_data/'+ suite_path_dicts[args.suite]
 
suite = TestSuite.from_file(suite_path)

if args.suite == 'sentiment':
    # pretrained BERT model on SST-2
    model_name_or_path = 'textattack/'+args.base_model+'-SST-2'
    model = SSTModel(model_name_or_path)
elif args.suite == 'qqp':
    # pretrained BERT model on QQP
    model_name_or_path = 'textattack/'+args.base_model+'-qqp'
    model = QQPModel(model_name_or_path)
else:
    quit()
threshold = args.threshold
ppl = Perplexity()

coverage = neuronMultiSectionCoverage(test_model = model_name_or_path, max_seq_len = args.max_seq_len, 
                                    bins_word = args.bins_word, bins_attention = args.bins_attention, bz = 48,
                                    pre_limits = False, word_mask = True)
print('initializing from training data')
if args.mask:
    vocab = []
    vocab_file = open("selected_words.txt", "r")
    content_list = vocab_file.readlines()
    for a in content_list:
        vocab.append(a.strip('\n'))
trainset_masks = []
if args.suite == 'sentiment':
    text_key = 'sentence'

    trainset = HuggingFaceDataset('glue', 'sst2', 'train', shuffle = True)
    validset = HuggingFaceDataset('glue', 'sst2', 'validation', shuffle = True)
trainset_str = []
validset_str = []



for example in trainset:
    current_example = example[0][text_key]
    if args.mask:
        current_example = [word for word in current_example if word in selected_vocab]

    trainset_str.append(current_example)
for example in validset:
    current_example = example[0][text_key]
    if args.mask:
        current_example = [word for word in current_example if word in selected_vocab]

    validset_str.append(current_example)
#testset = HuggingFaceDataset('glue', 'sst2', args.split, shuffle = False)
if args.debug == 1:
    trainset_str = trainset_str[0:1000]

for example in trainset_str:
    
    trainset_masks.append([1 for i in range(128)] )
# initialize coverage from training set


save_coverage_init_file = os.path.join( './coverage_results/', args.base_model +'_'+args.suite+'_BW_'+ str(args.bins_word) + \
    '_BA_' + str(args.bins_attention) + '_INIT_' + str(len(trainset_str))+'.pkl')

if not os.path.exists(save_coverage_init_file):
    print('can\'t find!: ', save_coverage_init_file)
    coverage.initialize_from_training_dataset(trainset_str, trainset_masks, bz = 128)
    initial_coverage = coverage(trainset_str, trainset_masks, bz = 128) 
    
    pickle.dump(coverage, open(save_coverage_init_file, 'wb'))
else:
    print('*'*100)
    print('exists!' , save_coverage_init_file)
    print('*'*100)
    coverage = pickle.load(open(save_coverage_init_file, 'rb'))
    initial_coverage = coverage._compute_coverage() 
for test in suite.tests:
    
    if not args.specify_test:
        args.test_name = suite.tests[test].name
    if TYPE_MAP[type(suite.tests[test])] == args.type.upper() and suite.tests[test].name == args.test_name:
        if args.query_tests_only:
            print(suite.tests[test].name)
            continue

        if args.type.upper() == 'MFT' and (suite.tests[test].labels) is None:
            continue
        input_examples_orig = suite.tests[test].data
        input_examples = []

        test_examples = (suite.tests[test].to_raw_examples()) 
        #print(test_examples)
        shuffled_indices = list(range(len(test_examples)))  
        if args.query_subset_only:
            shuffled_indices = shuffled_indices[0:args.subset]

        #random.shuffle(shuffled_indices)
        test_examples = [test_examples[t] for t in shuffled_indices]
        test_indices = [suite.tests[test].example_list_and_indices()[1][t] for t in shuffled_indices]
        
        for initial, final in zip(test_indices, test_examples):
                if type(input_examples_orig[initial]) is not list:
                    input_examples_orig[initial] = [input_examples_orig[initial]]
                input_examples.append(input_examples_orig[initial][0])
        
        #print('*'*100)
        #print([a for a in input_examples])
        #print('*'*100)
        labels = suite.tests[test].labels
        if args.type.upper() == 'MFT' and type(suite.tests[test].labels) is not list:
                labels = [labels]*len(test_examples)
        
        # coverage filtering 
        original_number_test_examples = len(test_examples)
        with open(args.save_str + suite.tests[test].name +"_skipped_examples_"+str(len(trainset_str))+".txt", "rb") as fp:
            skipped_examples_list = pickle.load(fp)
        with open(args.save_str + suite.tests[test].name +"_selected_examples_"+str(len(trainset_str))+".txt", "rb") as fp:
            test_examples_list = pickle.load(fp)
        print([example[1] for example in test_examples_list])
        new_ppl = ppl.calc_ppl([example[1] for example in test_examples_list])[0]
        orig_ppl = ppl.calc_ppl([example[1] for example in (test_examples_list + skipped_examples_list)] )[0]
        #test_examples = [test_examples[i] for i in relevant_idxs]
        #if args.type.upper() == 'MFT': labels = [labels[i] for i in relevant_idxs]
        #input_examples = [input_examples[i] for i in relevant_idxs]

        
            
        print(f'{suite.tests[test].name}, {orig_ppl} ,{new_ppl} %')
           
            
        

    
    
    