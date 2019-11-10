# sample test with pytest

def func(x):
    return x + 1

def test_answer():
    # fail test
    # assert func(3) == 5
    
    # or pass test
    assert func(4) == 5


import textattack

def test_model(model, dataset):
    logits = self.model(ids)


num_examples = 1000
def test_all_models:
    # @TODO take advantage of the parallelization here AND
    # *at least* do inference in batch.
    for dataset_name in textattack.run_attack.MODELS_BY_DATASET:
        dataset = textattack.run_attack.DATASET_CLASS_NAMES[dataset_name](num_examples)
        model_names = textattack.run_attack.MODELS_BY_DATASET[dataset_name]
        for model_name in model_names:
            model = textattack.run_attack.MODEL_CLASS_NAMES[model_name]()
            print('Testing {model_name} on {dataset_name}...')
            test_model(dataset, model)
            for label, text in dataset: