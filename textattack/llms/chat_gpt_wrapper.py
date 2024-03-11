import os

from textattack.models.wrappers import ModelWrapper


class ChatGptWrapper(ModelWrapper):
    """A wrapper around OpenAI's ChatGPT model. Note that you must provide your
    own API key to use this wrapper.

    Args:
        model_name (:obj:`str`): The name of the GPT model to use. See the OpenAI documentation
            for a list of latest model names
        key_environment_variable (:obj:`str`, 'optional`, defaults to :obj:`OPENAI_API_KEY`):
            The environment variable that the API key is set to
    """

    def __init__(
        self, model_name="gpt-3.5-turbo", key_environment_variable="OPENAI_API_KEY"
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv(key_environment_variable))

    def __call__(self, text_input_list):
        """Returns a list of responses to the given input list."""
        if isinstance(text_input_list, str):
            text_input_list = [text_input_list]

        outputs = []
        for text in text_input_list:
            completion = self.client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": text}]
            )
            outputs.append(completion.choices[0].message)

        return outputs
