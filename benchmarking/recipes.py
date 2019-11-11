import textattack
import time

from textattack.attack_recipes import Alzantot2018GeneticAlgorithm
from textattack.attack_recipes import Jin2019TextFooler

def _cb(s): return textattack.utils.color_text_terminal(str(s), color='blue')
def _cr(s): return textattack.utils.color_text_terminal(str(s), color='red')

def attack_dataset(attack, dataset):

    load_time = time.time()
    
    
    attack.attack(data, shuffle=False)

    finish_time = time.time()

    print(f'Loaded in {_cr(load_time - start_time)}s')
    print(f'Ran attack in {_cr(finish_time - load_time)}s')
    print(f'TOTAL TIME: {_cr(finish_time - start_time)}s')
    
def test_all_recipes(model, datasets):
    recipes = (
        ('Alzantot2018GeneticAlgorithm', Alzantot2018GeneticAlgorithm(model)),
        ('Jin2019TextFooler', Jin2019TextFooler(model))
    )
    
    for dataset_name, dataset in datasets:
        for recipe_name, recipe in recipes:
            test_attack(recipe, dataset)
            print(f'\nTesting {_cb(recipe_name)} on {_cb(dataset_name)}...')
            attack_dataset(recipe, dataset)
            print()
    print('-' * 60)

def test_all_models():
    model_names = textattack.run_attack.MODELS_BY_DATASET[dataset_name]
    for model_name in model_names:
        model = textattack.run_attack.MODEL_CLASS_NAMES[model_name]()
    
    model = models.classification.bert.BERTForIMDBSentimentClassification()
    
    datasets = (
        ('IMDBSentiment', textattack.datasets.classification.IMDBSentiment())
    )
    
    test_all_recipes(model, datasets)

if __name__ == '__main__': main()