import argparse
import os
import textattack
import sys
import time

from textattack.attack_recipes import Alzantot2018GeneticAlgorithm, Jin2019TextFooler
from textattack import models

def _cb(s): return textattack.utils.color_text_terminal(str(s), color='blue')
def _cg(s): return textattack.utils.color_text_terminal(str(s), color='green')
def _cr(s): return textattack.utils.color_text_terminal(str(s), color='red')


def test_attack(attack, dataset):
    load_time = time.time()
    attack.attack(dataset)
    finish_time = time.time()

    print(f'Loaded in {load_time - start_time}s')
    print(f'Ran attack in {finish_time - load_time}s')
    print(f'TOTAL TIME: {finish_time - start_time}s')
    
def test_all_recipes(recipes, model, datasets):
    for dataset_name, dataset in datasets:
        for recipe_class in recipes:
            recipe = recipe_class(model)
            print(f'\nTesting {_cb(recipe_class.__name__)} on {_cb(dataset_name)}...')
            test_attack(recipe, dataset)
            print()
    print(_cg('-') * 60)

def test_model(recipes, num_examples):
    model = models.classification.bert.BERTForIMDBSentimentClassification()
    
    datasets = (
        ('IMDBSentiment', textattack.datasets.classification.IMDBSentiment(num_examples)),
    )
    
    test_all_recipes(recipes, model, datasets)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', '--r', default=None, type=str,
        help="recipe to test (if you dont want to test them all)")
    parser.add_argument('--num_examples', '--n', type=int, default=100, 
        help="number of examples to test on")
    return parser.parse_args()

if __name__ == '__main__':
    # Only use one GPU, if we have one.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Disable tensorflow logs, except in the case of an error.
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    args = parse_args()
    
    # Store global start time.
    global start_time
    start_time = time.time()
    
    # Get recipe name param.
    recipe_name = args.recipe
    if recipe_name is None or recipe_name == 'all':
        recipes = [Alzantot2018GeneticAlgorithm, Jin2019TextFooler]
    elif recipe_name in ['alzantot', 'alzantot2018geneticalgorithm']:
        recipes = [Alzantot2018GeneticAlgorithm]
    elif recipe_name in ['textfooler', 'jin2019textfooler']:
        recipes = [Jin2019TextFooler]
    else:
        raise ValueError('Invalid recipe_name {args.recipe}')
    
    test_model(recipes, args.num_examples)