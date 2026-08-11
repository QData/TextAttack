"""
RemoteModelWrapper class
--------------------------

"""

import requests
import torch

from .model_wrapper import ModelWrapper


class RemoteModelWrapper(ModelWrapper):
    """This model wrapper queries a remote model with a list of text inputs.
    It sends each input to a remote HTTP endpoint provided in ``api_url``
    and parses the JSON response into class scores.

    Since the request and response format of a remote model is
    API-specific, ``request_fn`` and ``response_fn`` can be provided to
    adapt this wrapper to any endpoint. By default, the wrapper POSTs
    ``{"text": <input>}`` as a JSON body and expects a JSON response of
    the form ``{"negative": <score>, "positive": <score>}``.

    Args:
        api_url (:obj:`str`): The URL of the remote model's inference endpoint.
        request_fn (:obj:`Callable[[str], dict]`, `optional`): Builds the
            JSON request payload for a single piece of text. Defaults to
            ``lambda text: {"text": text}``.
        response_fn (:obj:`Callable[[dict], list]`, `optional`): Extracts a
            list of class scores from the parsed JSON response for a single
            input. Defaults to ``lambda result: [result["negative"], result["positive"]]``.
        timeout (:obj:`int`, `optional`, defaults to :obj:`10`): Per-request
            timeout, in seconds.

    Example::

        >>> import textattack

        >>> api_url = "https://example.com/predict"
        >>> model_wrapper = textattack.models.wrappers.RemoteModelWrapper(api_url)

        >>> attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
        >>> dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")
        >>> attack_args = textattack.AttackArgs(num_examples=100)
        >>> attacker = textattack.Attacker(attack, dataset, attack_args)
        >>> attacker.attack_dataset()
    """

    def __init__(self, api_url, request_fn=None, response_fn=None, timeout=10):
        self.api_url = api_url
        self.timeout = timeout
        self.request_fn = request_fn or (lambda text: {"text": text})
        self.response_fn = response_fn or (
            lambda result: [result["negative"], result["positive"]]
        )
        # `RemoteModelWrapper` has no local model: the remote endpoint is the
        # model. `GoalFunction` only uses this to check task compatibility,
        # and gracefully warns (rather than erroring) when it's unrecognized.
        self.model = None

    def __call__(self, text_input_list):
        predictions = []
        for text in text_input_list:
            response = requests.post(
                self.api_url, json=self.request_fn(text), timeout=self.timeout
            )
            if response.status_code != 200:
                raise ValueError(
                    f"API call failed with status {response.status_code}: {response.text}"
                )
            predictions.append(self.response_fn(response.json()))
        return torch.tensor(predictions)
