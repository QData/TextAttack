"""

python measure_coverage.py --split test --ratio 0.001 --coverage attention --length 3 --prespecify-limits  --attack textwordbug --test-ratio 5



"""

import torch
import os
import textattack
import copy
import pickle
from textattack.models.tokenizers import AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.models.wrappers import ModelWrapper
from transformers import AutoModelForSequenceClassification
from textattack.coverage import neuronMultiSectionCoverage
from textattack.augmentation import Augmenter
from textattack.attack_results import SuccessfulAttackResult
from textattack.datasets import HuggingFaceDataset
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.attack_recipes import TextFoolerJin2019, HotFlipEbrahimi2017, DeepWordBugGao2018, FasterGeneticAlgorithmJia2019, BAEGarg2019
from math import floor
import random
import numpy as np
import argparse

def random_seed(seed):
		torch.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)
		return

DEFAULT_CONSTRAINTS = [textattack.constraints.pre_transformation.RepeatModification(), textattack.constraints.pre_transformation.StopwordModification()]
available_transformations = [
		textattack.transformations.WordDeletion,
		textattack.transformations.RandomSynonymInsertion,
		textattack.transformations.WordSwapEmbedding,
		textattack.transformations.WordSwapChangeLocation,
		textattack.transformations.WordSwapChangeName,
		textattack.transformations.WordSwapChangeNumber,
		textattack.transformations.WordSwapContract,
		textattack.transformations.WordSwapExtend,
		textattack.transformations.WordSwapHomoglyphSwap,
		textattack.transformations.WordSwapMaskedLM,
		textattack.transformations.WordSwapQWERTY,
		textattack.transformations.WordSwapNeighboringCharacterSwap,
		textattack.transformations.WordSwapRandomCharacterDeletion,
		textattack.transformations.WordSwapRandomCharacterInsertion,
		textattack.transformations.WordSwapRandomCharacterSubstitution,
		textattack.transformations.RandomSwap,
		textattack.transformations.WordSwapWordNet
		]
random_seed(1)
parser = argparse.ArgumentParser(description='Measure Coverage of pretrained NLP Models')
parser.add_argument('--seed', type=int, default=1, help='set random seed')
parser.add_argument('--length', type=int, default=128, help='set max seq length')
parser.add_argument('--bins', type=int, default=10, help='set number of	bins/sections')
parser.add_argument('--ratio', type=float, default=1, help='proportion of train set used for dataset sampling')
parser.add_argument('--use-threshold', type=float, default=0.6, help='proportion of train set used for dataset sampling')
parser.add_argument('--test-ratio', type=int, default=1.0, help='proportion of train set used for dataset sampling')
parser.add_argument('--pct-words-to-swap', type=int, default=0.1, help='proportion of train set used for dataset sampling')
parser.add_argument('--dataset', type=str, default='imdb', help='dataset to use for measuring coverage')
parser.add_argument('--save-dir', type=str, default='./coverage/', help='dataset to use for measuring coverage')
parser.add_argument('--model', type=str, default='bert-base-uncased', help='model f whose weights to use')
parser.add_argument('--coverage', type=str, default='attention', help='coverage type')
# takes as input a transformation and a constraint
parser.add_argument('--transformation', type=int, default=0, help='transformation type')
parser.add_argument('--constraint', type=str, default='none', help='constraint type')

parser.add_argument('--split', type=str, default='test', help='split to use for measuring coverage')
parser.add_argument('--base-only', action='store_true', help='loading only base model')
parser.add_argument('--prespecify-limits', action='store_true', help='prespecify')
args = parser.parse_args()
random_seed(args.seed)



if not args.base_only:
	if args.dataset == 'sst2': 
		test_model = 'textattack/' + str(args.model) + '-' + 'SST-2'
	elif args.dataset == 'rotten-tomatoes': 
		test_model = 'textattack/' + str(args.model) + '-' + 'rotten_tomatoes'
	else:
		test_model = 'textattack/' + str(args.model) + '-' + str(args.dataset)
else:
	test_model = args.model
text_key = 'text'
# test_model="textattack/bert-base-uncased-ag-news",																																																																							
if args.dataset == 'sst2':
	text_key = 'sentence'
	trainset = HuggingFaceDataset('glue', 'sst2', 'train', shuffle = True)
	testset = HuggingFaceDataset('glue', 'sst2', args.split, shuffle = True)
elif args.dataset == 'rotten-tomatoes':
	trainset = HuggingFaceDataset('rotten_tomatoes', None, 'train', shuffle = True)
	testset = HuggingFaceDataset('rotten_tomatoes', None, args.split, shuffle = True)
else:
	trainset = HuggingFaceDataset(args.dataset, None, 'train', shuffle = True)
	testset = HuggingFaceDataset(args.dataset, None, args.split, shuffle = True)
	


if args.ratio <= 1.0:
	trainset = trainset[0:floor(args.ratio*len(trainset))]
else:
	trainset = trainset[0:floor(args.ratio)]


trainset_str = []
for example in trainset:
	
	trainset_str.append(example[0][text_key])

if args.test_ratio <= 1.0:
	testset = testset[0:floor(args.test_ratio*len(testset))]
else:
	testset = testset[0:floor(args.test_ratio)]

testset_str = []
for example in testset:
	testset_str.append(example[0][text_key])


args.save_dir += 'Sanity_COVER_' + args.coverage + '/'
os.makedirs(args.save_dir, exist_ok = True)
args.save_dir += 'SEED_'+str(args.seed) + '_BINS_' + str(args.bins) + '/'
os.makedirs(args.save_dir, exist_ok = True)
args.save_dir += 'data_' + str(args.dataset) + '_model_' + str(args.model) + '_ratio_' + str(args.ratio) + '_test_ratio_' + str(args.test_ratio) +'_L_'+ str(args.length)  + '_B_' + str(args.base_only) + '/'


os.makedirs(args.save_dir, exist_ok = True)
args.save_dir += 'transformation_' + str(args.transformation) + '_limits_' + str(args.prespecify_limits)
os.makedirs(args.save_dir, exist_ok = True)

# make coverage object
coverage = neuronMultiSectionCoverage(test_model = test_model, max_seq_len = args.length, k_m = args.bins, coverage = (args.coverage), pre_limits = (not (args.coverage == 'word') and args.prespecify_limits))

print('initializing from training data')
coverage.initialize_from_training_dataset(trainset_str)

print('--'*50)
print('generating test set!')
print('--'*50)
num_successes = 0.0
total = 1.0
if args.transformation != -1:
	if args.constraint != 'use':
		constraints = DEFAULT_CONSTRAINTS + [(UniversalSentenceEncoder(threshold=args.use_threshold))]
	else:
		constraints = DEFAULT_CONSTRAINTS
	augment_using_tf = Augmenter(transformation=available_transformations[args.transformation](),constraints=constraints,pct_words_to_swap=args.pct_words_to_swap,transformations_per_example=1,)
	#augment_using_tf.augment()[0]
	new_text = []	
	for text in new_text:
		new_text += augment_using_tf.augment(text )[0]




augmented_text_file = open(os.path.join(args.save_dir, 'examples.txt'), 'w')
pattern_text_file = open(os.path.join(args.save_dir, 'pattern.txt'), 'w')
for test in testset_str:
	augmented_text_file.write(test+'\n')
augmented_text_file.write('\n')
for test in new_text:
	augmented_text_file.write(test+'\n')
augmented_text_file.close()



# get pattern independent of the other examples in the test set 
# pattern for each example 

for test_example in testset_str:
	print(test_example)
	temp_coverage = copy.deepcopy(coverage)
	coverage_vector = temp_coverage.vector(test_example)
	# coverage_vector is a list
	del temp_coverage
	pattern_text_file.write(' '.join([str(i) for i in coverage_vector])+'\n')
	del coverage_vector
pattern_text_file.write('\n')
	

# also get the same for each augmented example 
for test_example in new_text:
	temp_coverage = copy.deepcopy(coverage)
	coverage_vector = temp_coverage.vector(test_example)
	# coverage_vector is a list
	pattern_text_file.write(' '.join([str(i) for i in coverage_vector])+'\n')
	

	del temp_coverage
# save the results too

pattern_text_file.close()






