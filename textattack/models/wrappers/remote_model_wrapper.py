"""
RemoteModelWrapper class
--------------------------

"""

import requests
import torch
import numpy as np
import transformers

class RemoteModelWrapper():
    """This model wrapper queries a remote model with a list of text inputs.  It sends the input to a remote endpoint provided in api_url.

    Args:
        api_url (:obj:`<TYPE HERE>`): <DESCRIPTION HERE>
    """
    def __init__(self, api_url):
        self.api_url = api_url
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")

    def __call__(self, text_input_list):
        predictions = []
        for text in text_input_list:
            params = dict()
            params["text"] = text
            response = requests.post(self.api_url, params=params, timeout=10)  # Use POST with JSON payload
            if response.status_code != 200:
                print(f"Response content: {response.text}")
                raise ValueError(f"API call failed with status {response.status_code}")
            result = response.json()
            # Assuming the API returns probabilities for positive and negative
            predictions.append([result["negative"], result["positive"]])
        return torch.tensor(predictions)

"""
Example usage: 

    >>> # Define the remote model API endpoint
    >>> api_url = "https://example.com"

    >>> model_wrapper = RemoteModelWrapper(api_url)

    >>> # Build the attack
    >>> attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

    >>> # Define dataset and attack arguments
    >>> dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

    >>> attack_args = textattack.AttackArgs(
    ...     num_examples=100,
    ...     log_to_csv="/textfooler.csv",
    ...     checkpoint_interval=5,
    ...     checkpoint_dir="checkpoints", 
    ...     disable_stdout=True
    ... )

    >>> # Run the attack
    >>> attacker = textattack.Attacker(attack, dataset, attack_args)
    >>> attacker.attack_dataset()
"""