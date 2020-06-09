import textattack
import torch

def model_predict(model, ids):
    import pdb; pdb.set_trace()
    id_dim = get_list_dim(ids)
    
    if id_dim == 2:
        # For models where the input is a single vector.
        ids = pad_lists(ids)
        ids = torch.tensor(ids).to(textattack.shared.utils.device)
        outputs = model(ids)
    elif id_dim == 3:
        # For models that take multiple vectors per input.
        ids = map(list, zip(*ids))
        ids = map(pad_lists, ids)
        ids = (torch.tensor(x).to(textattack.shared.utils.device) for x in ids)
    else:
        raise TypeError(f'Error: malformed ids.ndim ({id_dim})')
    
    outputs = model(*ids)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return outputs

def get_list_dim(ids):
    if isinstance(ids, tuple) or isinstance(ids, list) or isinstance(ids, torch.Tensor):
        return 1 + get_list_dim(ids[0])
    else:
        return 0

def pad_lists(lists, pad_token=0):
    """ Pads lists with trailing zeros to make them all the same length. """
    max_list_len = max(len(l) for l in lists)
    for i in range(len(lists)):
        lists[i] += ([pad_token] * (max_list_len - len(lists[i])))
    return lists