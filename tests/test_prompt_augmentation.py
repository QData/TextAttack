def test_prompt_augmentation_pipeline():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from textattack.augmentation.recipes import CheckListAugmenter
    from textattack.constraints.pre_transformation import UnmodifiableIndices
    from textattack.llms import HuggingFaceLLMWrapper
    from textattack.prompt_augmentation import PromptAugmentationPipeline

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model_wrapper = HuggingFaceLLMWrapper(model, tokenizer)

    augmenter = CheckListAugmenter()

    pipeline = PromptAugmentationPipeline(augmenter, model_wrapper)

    prompt = "As a sentiment classifier, determine whether the following text is 'positive' or 'negative'. Please classify: Poor Ben Bratt couldn't find stardom if MapQuest emailed him point-to-point driving directions."
    prompt_constraints = [UnmodifiableIndices([2, 3, 10, 12, 14])]

    output = pipeline(prompt, prompt_constraints)

    assert len(output) == 1
    assert len(output[0]) == 2
    assert "could not" in output[0][0]
    assert "negative" in output[0][1]
