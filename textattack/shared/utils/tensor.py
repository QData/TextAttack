import torch

import textattack
from textattack.shared import utils


def batch_tokenize(tokenizer, attacked_text_list):
    """ Tokenizes a list of inputs and returns their tokenized forms in a list. """
    inputs = [at.tokenizer_input for at in attacked_text_list]
    if hasattr(tokenizer, "batch_encode"):
        return tokenizer.batch_encode(inputs)
    else:
        return [tokenizer.encode(x) for x in inputs]


def batch_model_predict(model, inputs, batch_size=32):
    outputs = []
    i = 0
    while i < len(inputs):
        batch = inputs[i : i + batch_size]
        batch_preds = model_predict(model, batch)
        outputs.append(batch_preds)
        i += batch_size
    try:
        return torch.cat(outputs, dim=0)
    except TypeError:
        # A TypeError occurs when the lists in ``outputs`` are full of strings
        # instead of numbers. If this is the case, just return the regular
        # list.
        return outputs


def get_model_device(model):
    if hasattr(model, "model"):
        model_device = next(model.model.parameters()).device
    else:
        model_device = next(model.parameters()).device
    return model_device


def model_predict(model, inputs):
    try:
        return try_model_predict(model, inputs)
    except Exception as e:
        textattack.shared.utils.logger.error(
            f"Failed to predict with model {model.__class__}. Check tokenizer configuration."
        )
        raise e


def try_model_predict(model, inputs):
    model_device = get_model_device(model)

    if isinstance(inputs, torch.Tensor):
        # If `inputs` is a tensor, we'll assume it's been pre-processed, and send
        # it to the model.
        if model_device != inputs.device:
            inputs = inputs.to(model_device)
        outputs = model(inputs)

    elif isinstance(inputs, dict):
        # If `inputs` is a single dict, its values are assumed to be input
        # tensors.
        outputs = model(**inputs)

    elif isinstance(inputs[0], dict):
        # If ``inputs`` is a list of dicts, we convert them to a single dict
        # (now of tensors) and pass to the model as kwargs.
        # Convert list of dicts to dict of lists.
        input_dict = {k: [_dict[k] for _dict in inputs] for k in inputs[0]}
        # Convert list keys to tensors.
        for key in input_dict:
            input_dict[key] = pad_lists(input_dict[key])
            input_dict[key] = torch.tensor(input_dict[key]).to(model_device)
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
            inputs = torch.tensor(inputs).to(model_device)
            outputs = model(inputs)
        elif input_dim == 3:
            # For models that take multiple vectors per input.
            inputs = map(list, zip(*inputs))
            inputs = map(pad_lists, inputs)
            inputs = (torch.tensor(x).to(model_device) for x in inputs)
            outputs = model(*inputs)
        else:
            raise TypeError(f"Error: malformed inputs.ndim ({input_dim})")

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
        lists[i] += [pad_token] * (max_list_len - len(lists[i]))
    return lists
