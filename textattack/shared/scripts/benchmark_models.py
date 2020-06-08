import argparse
import textattack
import torch
import sys

from attack_args_helper import get_args, parse_model_from_args, parse_dataset_from_args

def _cb(s): return textattack.shared.utils.color_text(str(s), color='blue', method='ansi')
def _cg(s): return textattack.shared.utils.color_text(str(s), color='green', method='ansi')
def _cr(s): return textattack.shared.utils.color_text(str(s), color='red', method='ansi')
def _pb(): print(_cg('-' * 60))

from collections import Counter

def get_num_successes(args, model, ids, true_labels):
    id_dim = torch.tensor(ids).ndim
    if id_dim == 2:
        # For models where the input is a single vector.
        ids = textattack.shared.utils.preprocess_ids(ids)
        ids = torch.tensor(ids).to(textattack.shared.utils.device)
        preds = model(ids)
    elif id_dim == 3:
        # For models that take multiple vectors per input.
        ids = map(torch.tensor, zip(*ids))
        ids = (x.to(textattack.shared.utils.device) for x in ids)
        preds = model(*ids)
    else:
        raise TypeError(f'Error: malformed id_dim ({id_dim})')
    true_labels = torch.tensor(true_labels).to(textattack.shared.utils.device)
    if isinstance(preds, tuple):
        preds = preds[0]
    guess_labels = preds.argmax(dim=1)
    successes = (guess_labels == true_labels).sum().item()
    return successes, true_labels, guess_labels

def test_model_on_dataset(args, model, dataset, batch_size=16):
    num_examples = args.num_examples
    succ = 0
    fail = 0
    batch_ids = []
    batch_labels = []
    all_true_labels = []
    all_guess_labels = []
    for i, (text, label) in enumerate(dataset):
        if i >= num_examples: break
        ids = model.tokenizer.encode(text)
        batch_ids.append(ids)
        batch_labels.append(label)
        if len(batch_ids) == batch_size:
            batch_succ, true_labels, guess_labels = get_num_successes(args, model, batch_ids, batch_labels)
            # for t, g in zip(true_labels, guess_labels):
                # print(t, 'but guessed', g)
            batch_fail = batch_size - batch_succ
            succ += batch_succ
            fail += batch_fail
            batch_ids = []
            batch_labels = []
            all_true_labels.extend(true_labels.tolist())
            all_guess_labels.extend(guess_labels.tolist())
    if len(batch_ids) > 0:
        batch_succ, true_labels, guess_labels = get_num_successes(args, model, batch_ids, batch_labels)
        batch_fail = len(batch_ids) - batch_succ
        succ += batch_succ
        fail += batch_fail
        all_true_labels.extend(true_labels.tolist())
        all_guess_labels.extend(guess_labels.tolist())
    
    perc = float(succ)/(succ+fail)*100.0
    perc = '{:.2f}%'.format(perc)
    print(f'Successes {succ}/{succ+fail} ({_cb(perc)})')
    return perc

if __name__ == '__main__': 
    args = get_args()
    
    model = parse_model_from_args(args)
    dataset = parse_dataset_from_args(args)
    
    with torch.no_grad():
        test_model_on_dataset(args, model, dataset)