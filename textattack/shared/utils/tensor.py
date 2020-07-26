import torch


def batch_model_predict(model_predict, inputs, batch_size=32):
    """Runs prediction on iterable ``inputs`` using batch size ``batch_size``.

    Aggregates all predictions into an ``np.ndarray``.
    """
    outputs = []
    i = 0
    while i < len(inputs):
        batch = inputs[i : i + batch_size]
        batch_preds = model_predict(batch)
        if not isinstance(batch_preds, torch.Tensor):
            batch_preds = torch.Tensor(batch_preds)
        outputs.append(batch_preds)
        i += batch_size

    return torch.cat(outputs, dim=0)
