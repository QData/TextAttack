import textattack
import torch

def model_predict(model, inputs):
    try:
        return try_model_predict(model, inputs)
    except Exception as e:
        textattack.shared.utils.logger.warn(f'Failed to predict with model {model}. Check tokenizer configuration.')
        raise e

def try_model_predict(model, inputs):
    if isinstance(inputs[0], dict):
        # If ``inputs`` is a list of dicts, we convert them to a single dict
        # (now of tensors) and pass to the model as kwargs.
        # Convert list of dicts to dict of lists.
        input_dict = {k: [_dict[k] for _dict in inputs] for k in inputs[0]}
        # Convert list keys to tensors.
        for key in input_dict:
            input_dict[key] = pad_lists(input_dict[key])
            input_dict[key] = torch.tensor(input_dict[key]).to(textattack.shared.utils.device)
        # Do inference using keys as kwargs.
        outputs = model(**input_dict)
        
    else:
        # If ``inputs`` is not a list of dicts, it's either a list of tuples
        # (model takes multiple inputs) or a list of ID lists (where the model
        # takes a single input). In this case, we'll do our best to figure out
        # the proper input to the model, anyway.
        input_dim = get_list_dim(inputs)
        
        if input_dim == 2:
            # For models where the input is a single vector.
            inputs = pad_lists(inputs)
            inputs = torch.tensor(inputs).to(textattack.shared.utils.device)
            outputs = model(inputs)
        elif input_dim == 3:
            # For models that take multiple vectors per input.
            inputs = map(list, zip(*inputs))
            inputs = map(pad_lists, inputs)
            inputs = (torch.tensor(x).to(textattack.shared.utils.device) for x in inputs)
            outputs = model(*inputs)
        else:
            raise TypeError(f'Error: malformed inputs.ndim ({input_dim})')
    
    # If `outputs` is a tuple, take the first argument.
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