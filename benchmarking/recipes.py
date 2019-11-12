import os
import textattack
import time

from textattack.attack_recipes import Alzantot2018GeneticAlgorithm, Jin2019TextFooler
from textattack import models

def _cb(s): return textattack.utils.color_text_terminal(str(s), color='blue')
def _cr(s): return textattack.utils.color_text_terminal(str(s), color='red')

def test_attack(attack, dataset):
    load_time = time.time()
    attack.attack(dataset)
    finish_time = time.time()

    print(f'Loaded in {_cr(load_time - start_time)}s')
    print(f'Ran attack in {_cr(finish_time - load_time)}s')
    print(f'TOTAL TIME: {_cr(finish_time - start_time)}s')
    
def test_all_recipes(model, datasets):
    recipes = (
        ('Alzantot2018GeneticAlgorithm', Alzantot2018GeneticAlgorithm(model)),
        ('Jin2019TextFooler', Jin2019TextFooler(model))
    )
    
    print('datasets:',datasets)
    
    for dataset_name, dataset in datasets:
        for recipe_name, recipe in recipes:
            print(f'\nTesting {_cb(recipe_name)} on {_cb(dataset_name)}...')
            test_attack(recipe, dataset)
            print()
    print('-' * 60)

def test_all_models():
    # model_names = textattack.run_attack.MODELS_BY_DATASET[dataset_name]
    # for model_name in model_names:
        # model = textattack.run_attack.MODEL_CLASS_NAMES[model_name]()
    
    model = models.classification.bert.BERTForIMDBSentimentClassification()
    
    datasets = (
        ('IMDBSentiment', textattack.datasets.classification.IMDBSentiment(10)),
    )
    
    test_all_recipes(model, datasets)

if __name__ == '__main__':
    # Only use one GPU, if we have one.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Disable tensorflow logs, except in the case of an error.
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    test_all_models()