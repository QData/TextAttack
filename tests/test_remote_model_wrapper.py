import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from textattack.models.wrappers import ModelWrapper, RemoteModelWrapper

# Set this to a live endpoint to also exercise RemoteModelWrapper against a
# real API (e.g. a local Ollama server's http://localhost:11434/api/generate).
# Left unset, the live test below is skipped and only the mocked
# https://example.com/predict cases run -- which is what CI does.
#
# To run the live test too:
#   REMOTE_MODEL_WRAPPER_TEST_URL=http://localhost:11434/api/generate \
#   REMOTE_MODEL_WRAPPER_TEST_MODEL=qwen2.5:7b-instruct \
#       pytest tests/test_remote_model_wrapper.py -v
#
# See CONTRIBUTING.md#tests for more on when/why this test is opt-in.
LIVE_API_URL_ENV_VAR = "REMOTE_MODEL_WRAPPER_TEST_URL"
LIVE_API_MODEL_ENV_VAR = "REMOTE_MODEL_WRAPPER_TEST_MODEL"


def _mock_response(status_code=200, json_body=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body or {}
    response.text = "error body"
    return response


def test_is_model_wrapper_subclass_with_no_local_model():
    wrapper = RemoteModelWrapper("https://example.com/predict")
    assert isinstance(wrapper, ModelWrapper)
    assert wrapper.model is None


def test_call_posts_json_body_and_parses_default_schema():
    wrapper = RemoteModelWrapper("https://example.com/predict")
    mock_post = MagicMock(
        return_value=_mock_response(json_body={"negative": 0.1, "positive": 0.9})
    )
    with patch(
        "textattack.models.wrappers.remote_model_wrapper.requests.post", mock_post
    ):
        result = wrapper(["great movie"])

    assert torch.allclose(result, torch.tensor([[0.1, 0.9]]))
    _, call_kwargs = mock_post.call_args
    assert call_kwargs["json"] == {"text": "great movie"}


def test_call_raises_on_non_200_status():
    wrapper = RemoteModelWrapper("https://example.com/predict")
    mock_post = MagicMock(return_value=_mock_response(status_code=500))
    with patch(
        "textattack.models.wrappers.remote_model_wrapper.requests.post", mock_post
    ):
        with pytest.raises(ValueError):
            wrapper(["great movie"])


def test_call_supports_custom_request_and_response_fn():
    wrapper = RemoteModelWrapper(
        "https://example.com/predict",
        request_fn=lambda text: {"inputs": text},
        response_fn=lambda body: body["scores"],
    )
    mock_post = MagicMock(
        return_value=_mock_response(json_body={"scores": [0.2, 0.3, 0.5]})
    )
    with patch(
        "textattack.models.wrappers.remote_model_wrapper.requests.post", mock_post
    ):
        result = wrapper(["some text"])

    assert torch.allclose(result, torch.tensor([[0.2, 0.3, 0.5]]))
    _, call_kwargs = mock_post.call_args
    assert call_kwargs["json"] == {"inputs": "some text"}


def test_call_sends_one_request_per_input_in_order():
    wrapper = RemoteModelWrapper("https://example.com/predict")
    responses = [
        _mock_response(json_body={"negative": 0.1, "positive": 0.9}),
        _mock_response(json_body={"negative": 0.8, "positive": 0.2}),
    ]
    mock_post = MagicMock(side_effect=responses)
    with patch(
        "textattack.models.wrappers.remote_model_wrapper.requests.post", mock_post
    ):
        result = wrapper(["great movie", "terrible movie"])

    assert torch.allclose(result, torch.tensor([[0.1, 0.9], [0.8, 0.2]]))
    sent_texts = [call.kwargs["json"]["text"] for call in mock_post.call_args_list]
    assert sent_texts == ["great movie", "terrible movie"]


def test_call_passes_custom_timeout_to_requests():
    wrapper = RemoteModelWrapper("https://example.com/predict", timeout=3)
    mock_post = MagicMock(
        return_value=_mock_response(json_body={"negative": 0.1, "positive": 0.9})
    )
    with patch(
        "textattack.models.wrappers.remote_model_wrapper.requests.post", mock_post
    ):
        wrapper(["great movie"])

    _, call_kwargs = mock_post.call_args
    assert call_kwargs["timeout"] == 3


@pytest.mark.skipif(
    not os.environ.get(LIVE_API_URL_ENV_VAR),
    reason=(
        f"Set {LIVE_API_URL_ENV_VAR} to a live endpoint to run "
        "RemoteModelWrapper against a real API (not run in CI)."
    ),
)
def test_call_against_real_user_provided_url():
    """Live sanity check: hits an actual HTTP endpoint, no mocking.

    Example (a local Ollama server)::

        REMOTE_MODEL_WRAPPER_TEST_URL=http://localhost:11434/api/generate \\
        REMOTE_MODEL_WRAPPER_TEST_MODEL=qwen2.5:7b-instruct \\
            pytest tests/test_remote_model_wrapper.py -k real_user_provided_url -v
    """
    api_url = os.environ[LIVE_API_URL_ENV_VAR]

    if "/api/generate" in api_url:
        # Ollama's generate endpoint: prompt in, generated text out. Adapt it
        # to a sentiment classifier via prompting, same as the demo script.
        model_name = os.environ.get(LIVE_API_MODEL_ENV_VAR, "qwen2.5:7b-instruct")

        def request_fn(text):
            prompt = (
                "Classify the sentiment of this movie review as exactly one "
                f'word, "positive" or "negative".\n\nReview: {text}\nSentiment:'
            )
            return {"model": model_name, "prompt": prompt, "stream": False}

        def response_fn(body):
            label = body["response"].strip().lower()
            return [0.0, 1.0] if "positive" in label else [1.0, 0.0]

        wrapper = RemoteModelWrapper(
            api_url, request_fn=request_fn, response_fn=response_fn, timeout=60
        )
    else:
        # Assume the endpoint implements the wrapper's default schema:
        # {"text": ...} in, {"negative": ..., "positive": ...} out.
        wrapper = RemoteModelWrapper(api_url, timeout=60)

    result = wrapper(
        ["This movie was an absolute masterpiece, I loved every minute of it."]
    )
    assert result.shape == (1, 2)
    assert torch.argmax(result[0]).item() == 1  # positive
