def test_generate_default_max_length_omitted():
    # Regression test: the `.generate()` branch added for raw encoder-
    # decoder generation models (#771) used to pass no length control at
    # all, always falling back to whatever transformers/the model's own
    # generation_config decided. `max_length=None` (the default) should
    # leave `max_length` out of the `.generate()` call entirely, so a
    # checkpoint with its own sensible generation_config isn't overridden.
    import transformers

    from textattack.models.wrappers import HuggingFaceModelWrapper

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        "hf-internal-testing/tiny-random-t5"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-t5"
    )
    wrapper = HuggingFaceModelWrapper(model, tokenizer)

    captured = {}
    original_generate = model.generate

    def spy_generate(*args, **kwargs):
        captured.update(kwargs)
        return original_generate(*args, **kwargs)

    model.generate = spy_generate
    wrapper(["hello world"])

    assert "max_length" not in captured


def test_generate_explicit_max_length_passed_through():
    import transformers

    from textattack.models.wrappers import HuggingFaceModelWrapper

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        "hf-internal-testing/tiny-random-t5"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-t5"
    )
    wrapper = HuggingFaceModelWrapper(model, tokenizer, max_length=7)

    captured = {}
    original_generate = model.generate

    def spy_generate(*args, **kwargs):
        captured.update(kwargs)
        return original_generate(*args, **kwargs)

    model.generate = spy_generate
    wrapper(["hello world"])

    assert captured.get("max_length") == 7


def test_generation_model_routes_to_generate():
    import transformers

    from textattack.models.wrappers import HuggingFaceModelWrapper

    model = transformers.BartForConditionalGeneration.from_pretrained(
        "hf-internal-testing/tiny-random-bart"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-bart"
    )
    tokenizer.model_max_length = 16
    wrapper = HuggingFaceModelWrapper(model, tokenizer)

    output = wrapper(["hello world"])

    assert isinstance(output, list)
    assert all(isinstance(o, str) for o in output)


def test_classification_model_on_encoder_decoder_backbone_routes_to_logits():
    # Regression test: routing into `.generate()` used to be decided by
    # `hasattr(self.model, "generate")` alone, which on older transformers
    # versions was true for every `PreTrainedModel` regardless of whether
    # it actually had a generation-capable head - risking misrouting a
    # seq2seq-backbone classification model (e.g. BartForSequenceClassification,
    # whose config also sets is_encoder_decoder=True) into `.generate()`.
    # `can_generate()` correctly says no for this model on the currently
    # pinned transformers version; this locks in that the classification
    # path (plain forward pass -> logits) is what actually gets used.
    import transformers

    from textattack.models.wrappers import HuggingFaceModelWrapper

    model = transformers.BartForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-bart"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-bart"
    )
    tokenizer.model_max_length = 16
    wrapper = HuggingFaceModelWrapper(model, tokenizer)

    output = wrapper(["hello world"])

    assert output.shape == (1, model.config.num_labels)


def test_can_generate_preferred_over_hasattr_generate():
    # Regression test for the actual failure mode `can_generate()`
    # fixes: on transformers versions predating `can_generate()`,
    # `.generate` was defined on every `PreTrainedModel` regardless of
    # whether it had a generation-capable head, so `hasattr(model,
    # "generate")` alone couldn't distinguish a real generation model
    # from a seq2seq-backbone classification model. Simulate that by
    # attaching a `.generate` attribute directly (this classification
    # model doesn't have one on the currently pinned transformers
    # version) while `can_generate()` still correctly reports False, and
    # confirm the wrapper still doesn't call it.
    import transformers

    from textattack.models.wrappers import HuggingFaceModelWrapper

    model = transformers.BartForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-bart"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-bart"
    )
    tokenizer.model_max_length = 16

    def should_not_be_called(*args, **kwargs):
        raise AssertionError(
            "`.generate()` should not be called for a classification model"
        )

    model.generate = should_not_be_called
    assert model.can_generate() is False

    wrapper = HuggingFaceModelWrapper(model, tokenizer)
    output = wrapper(["hello world"])

    assert output.shape == (1, model.config.num_labels)


def test_t5_for_text_to_text_still_works():
    # `T5ForTextToText` (TextAttack's own helper) has no `.config`
    # attribute at all; confirm the defensive `getattr(self.model,
    # "config", None)` lookup added for the generation-routing check
    # doesn't break this path.
    from textattack.models.helpers import T5ForTextToText
    from textattack.models.tokenizers import T5Tokenizer
    from textattack.models.wrappers import HuggingFaceModelWrapper

    model = T5ForTextToText("english_to_german")
    tokenizer = T5Tokenizer("english_to_german")
    wrapper = HuggingFaceModelWrapper(model, tokenizer)

    output = wrapper(["Hello, how are you?"])

    assert isinstance(output, list)
    assert len(output) == 1
    assert isinstance(output[0], str)
