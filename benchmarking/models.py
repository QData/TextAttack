import argparse
import textattack
import torch
import sys

import textattack.models as models

def _cb(s): return textattack.utils.color(str(s), color='blue', method='stdout')
def _cg(s): return textattack.utils.color(str(s), color='green', method='stdout')
def _cr(s): return textattack.utils.color(str(s), color='red', method='stdout')
def _pb(): print(_cg('-' * 60))

from collections import Counter

def get_num_successes(model, ids, true_labels):
    ids = map(torch.tensor, zip(*ids))
    ids = (x.to(textattack.utils.get_device()) for x in ids)
    true_labels = torch.tensor(true_labels).to(textattack.utils.get_device())
    preds = model(*ids)
    guess_labels = preds.argmax(dim=1)
    successes = (guess_labels == true_labels).sum().item()
    return successes, true_labels, guess_labels

def test_model_on_dataset(model, dataset, batch_size=16):
    succ = 0
    fail = 0
    batch_ids = []
    batch_labels = []
    all_true_labels = []
    all_guess_labels = []
    for label, text in dataset:
        ids = model.tokenizer.encode(text)
        batch_ids.append(ids)
        batch_labels.append(label)
        if len(batch_ids) == batch_size:
            batch_succ, true_labels, guess_labels = get_num_successes(model, batch_ids, batch_labels)
            batch_fail = batch_size - batch_succ
            succ += batch_succ
            fail += batch_fail
            batch_ids = []
            batch_labels = []
            all_true_labels.extend(true_labels.tolist())
            all_guess_labels.extend(guess_labels.tolist())
    if len(batch_ids) > 0:
        batch_succ, true_labels, guess_labels = get_num_successes(model, batch_ids, batch_labels)
        batch_fail = len(batch_ids) - batch_succ
        succ += batch_succ
        fail += batch_fail
        all_true_labels.extend(true_labels.tolist())
        all_guess_labels.extend(guess_labels.tolist())
    
    print('true_labels:', Counter(all_true_labels))
    print('guess_labels:', Counter(all_guess_labels))
    
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
    with torch.no_grad():
        test_all_models(args.n)