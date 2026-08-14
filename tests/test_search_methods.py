import collections


def test_truncate_words_to_sorts_before_slicing():
    # Regression test: `indices_to_order` comes from a `set` intersection
    # upstream (in `Transformation.__call__`), so its iteration order isn't
    # guaranteed to ascend by word position. `truncate_words_to` used to
    # slice it directly (`indices_to_order[:n]`), which could pick an
    # arbitrary N-element subset spanning the whole text instead of the
    # first N word positions - defeating the cost bound this option exists
    # for (e.g. for `wir_method="gradient"` against a model with a limited
    # context window).
    from textattack.search_methods import GreedyWordSwapWIR
    from textattack.shared import AttackedText

    text = AttackedText(" ".join(f"word{i}" for i in range(20)))
    search = GreedyWordSwapWIR(wir_method="unk", truncate_words_to=5)

    # Simulate indices arriving out of ascending order, as they can from a
    # set-derived source.
    unsorted_indices = [17, 3, 9, 0, 14, 6, 19, 1, 11, 2]
    search.get_indices_to_order = lambda t, **kw: (
        len(unsorted_indices),
        list(unsorted_indices),
    )

    class FakeResult:
        score = 0.0

    search.get_goal_results = lambda texts: ([FakeResult() for _ in texts], False)

    order, search_over = search._get_index_order(text)

    # The 5 *smallest* values (first 5 word positions), not just any 5.
    assert set(order.tolist()) == set(sorted(unsorted_indices)[:5])


def test_gradient_wir_truncation_bounds_get_grad_input():
    # Regression test: `truncate_words_to` only used to shorten the cheap
    # post-hoc index-scoring loop for `wir_method="gradient"`; the
    # expensive step (`get_grad`'s tokenize + forward + backward pass)
    # still ran over the full untruncated text. Confirm the text actually
    # fed to `get_grad` is bounded to the truncated word span.
    import transformers

    from textattack.models.wrappers import HuggingFaceModelWrapper
    from textattack.search_methods import GreedyWordSwapWIR
    from textattack.shared import AttackedText

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-bert"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-bert"
    )
    wrapper = HuggingFaceModelWrapper(model, tokenizer)

    long_text = " ".join(f"word{i}" for i in range(200))
    attacked_text = AttackedText(long_text)

    search = GreedyWordSwapWIR(wir_method="gradient", truncate_words_to=10)
    search.get_victim_model = lambda: wrapper
    search.get_indices_to_order = lambda t, **kw: (
        len(t.words),
        list(range(len(t.words))),
    )

    captured = []
    original_get_grad = wrapper.get_grad

    def spy_get_grad(text_input):
        captured.append(text_input)
        return original_get_grad(text_input)

    wrapper.get_grad = spy_get_grad

    order, search_over = search._get_index_order(attacked_text)

    assert len(captured) == 1
    assert len(captured[0].split()) == 10
    assert len(order) == 10


def test_gradient_wir_truncation_leaves_paired_input_untouched():
    # Paired inputs (e.g. premise/hypothesis) have a tuple `tokenizer_input`;
    # truncating that here would need to preserve the pair structure to
    # avoid breaking the tokenizer's dual-sequence encoding, which is out
    # of scope. Confirm they fall back to the full, untruncated pair.
    import transformers

    from textattack.models.wrappers import HuggingFaceModelWrapper
    from textattack.search_methods import GreedyWordSwapWIR
    from textattack.shared import AttackedText

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-bert"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-bert"
    )
    wrapper = HuggingFaceModelWrapper(model, tokenizer)

    pair = collections.OrderedDict(
        [("premise", "a big brown dog runs fast"), ("hypothesis", "the dog is running")]
    )
    attacked_text = AttackedText(pair)

    search = GreedyWordSwapWIR(wir_method="gradient", truncate_words_to=3)
    search.get_victim_model = lambda: wrapper
    search.get_indices_to_order = lambda t, **kw: (
        len(t.words),
        list(range(len(t.words))),
    )

    captured = []
    original_get_grad = wrapper.get_grad

    def spy_get_grad(text_input):
        captured.append(text_input)
        return original_get_grad(text_input)

    wrapper.get_grad = spy_get_grad

    search._get_index_order(attacked_text)

    assert captured[0] == ("a big brown dog runs fast", "the dog is running")
