import textattack
import torch
import sys

def _cb(s): return textattack.utils.color_text_terminal(s, color='blue')
def _cr(s): return textattack.utils.color_text_terminal(s, color='red')

def test_model_on_dataset(model, dataset):
    # TODO do inference in batch.
    succ = 0
    fail = 0
    for label, text in dataset:
        ids = model.convert_text_to_ids(text)
        ids = torch.tensor([ids]).to(textattack.utils.get_device())
        pred_score = model(ids).squeeze()
        pred_label = pred_score.argmax().item()
        if label==pred_label: succ += 1
        else: fail += 1
    perc = float(succ)/(succ+fail)*100.0
    perc = '{:.2f}%'.format(perc)
    print(f'Successes {succ}/{succ+fail} ({_cb(perc)})')
    return perc


def test_all_models(num_examples=1000):
    for dataset_name in textattack.run_attack.MODELS_BY_DATASET:
        dataset = textattack.run_attack.DATASET_CLASS_NAMES[dataset_name](num_examples)
        model_names = textattack.run_attack.MODELS_BY_DATASET[dataset_name]
        for model_name in model_names:
            model = textattack.run_attack.MODEL_CLASS_NAMES[model_name]()
            print(f'\nTesting {_cr(model_name)} on {_cr(dataset_name)}...')
            test_model_on_dataset(model, dataset)
            print()
        print('-' * 60)
    # @TODO print the grid of models/dataset names with results in a nice table :)

if __name__ == '__main__': 
    # If an argument was passed in, use that as the number of samples to test on.
    try:
        n = int(sys.argv[1])
    except IndexError:
        n = 100
    test_all_models(n)