import argparse
import textattack
import torch
import sys

import textattack.models as models

def _cb(s): return textattack.utils.color(str(s), color='blue', method='stdout')
def _cg(s): return textattack.utils.color(str(s), color='green', method='stdout')
def _cr(s): return textattack.utils.color(str(s), color='red', method='stdout')
def _pb(): print(_cg('-' * 60))

def test_model_on_dataset(model, dataset, batch_size=8):
    # TODO do inference in batch.
    succ = 0
    fail = 0
    preds = []
    for label, text in dataset:
        ids = model.tokenizer.encode(text)
        ids = torch.tensor([ids]).to(textattack.utils.get_device())
        pred_score = model(ids).argmax(dim=1)
        pred_label = pred_score.argmax().item()
        preds.append(pred_label)
        if label==pred_label: succ += 1
        else: fail += 1
    perc = float(succ)/(succ+fail)*100.0
    perc = '{:.2f}%'.format(perc)
    print(f'Successes {succ}/{succ+fail} ({_cb(perc)})')
    return perc

def test_all_models(num_examples):
    _pb()
    for model_name in textattack.run_attack.MODEL_CLASS_NAMES:
        model = eval(textattack.run_attack.MODEL_CLASS_NAMES[model_name])()
        dataset = textattack.run_attack.DATASET_BY_MODEL[model_name]()
        print(f'\nTesting {_cr(model_name)} on {_cr(type(dataset))}...')
        test_model_on_dataset(model, dataset)
        _pb()
    # @TODO print the grid of models/dataset names with results in a nice table :)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100, 
        help="number of examples to test on")
    return parser.parse_args()

if __name__ == '__main__': 
    args = parse_args()
    test_all_models(args.n)