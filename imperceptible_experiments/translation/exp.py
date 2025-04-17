import textattack

from imperceptible_experiments.translation.att import translation_attack
from textattack.goal_functions import MaximizeLevenshtein, MaximizeBleu
from textattack.search_methods import ImperceptibleDE
from textattack.transformations import WordSwapHomoglyphSwap

import transformers
from transformers import AutoTokenizer, FSMTModel, FSMTForConditionalGeneration

# import datasets
# from datasets import load_dataset

import torch
from bs4 import BeautifulSoup

import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)

# Model wrapper

model = torch.hub.load('pytorch/fairseq',
                        'transformer.wmt14.en-fr',
                        tokenizer='moses',
                        bpe='subword_nmt',
                        verbose=False).eval()

model_wrapper = textattack.models.wrappers.FairseqTranslationWrapper(model)

num_examples = 5

source = dict()
target = dict()
with open('newstest2014-fren-src.en.sgm', 'r') as f:
    source_doc = BeautifulSoup(f, 'html.parser')
with open('newstest2014-fren-ref.fr.sgm', 'r') as f:
    target_doc = BeautifulSoup(f, 'html.parser')
for doc in source_doc.find_all('doc'):
    source[str(doc['docid'])] = dict()
    for seg in doc.find_all('seg'):
        source[str(doc['docid'])][str(seg['id'])] = str(seg.string)
for docid, doc in source.items():
    target[docid] = dict()
    for segid in doc:
        node = target_doc.select_one(f'doc[docid="{docid}"] > seg[id="{segid}"]')
        target[docid][segid] = str(node.string)
source_list = []
target_list = []
for docid, doc in source.items():
    for segid, seg in doc.items():
        source_list.append({ docid: { segid: seg }})
source_list.sort(key=lambda x: len(str(list(list(x.values())[0].values())[0])))
source_list = source_list[:num_examples]
for example in source_list:
    for docid, doc in example.items():
        for segid, seg in doc.items():
            target_list.append({ docid: { segid: target[docid][segid] }})

# Attack params
goal_function = MaximizeBleu(model_wrapper)
constraints = []
transformation = WordSwapHomoglyphSwap()
search_method = ImperceptibleDE(
    popsize=10, 
    maxiter=5, 
    verbose=True,
    max_perturbs=1
)

translation_attack(
    goal_function=goal_function, 
    constraints=constraints, 
    transformation=transformation, 
    search_method=search_method,
    model_wrapper=model_wrapper,
    source=source_list,
    target=target_list,
    num_rows=num_examples,
    results_path="translation_homo_bleu.jsonl"
)
