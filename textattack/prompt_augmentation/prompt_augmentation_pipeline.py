from textattack.constraints import PreTransformationConstraint


class PromptAugmentationPipeline:
    """A prompt augmentation pipeline to augment a prompt and obtain the
    responses from a LLM on the augmented prompts.

    Args:
        augmenter (textattack.Augmenter): the augmenter to use to
            augment the prompt
        llm (textattack.ModelWrapper): the LLM to generate responses
            to the augmented data
    """

    def __init__(self, augmenter, llm):
        self.augmenter = augmenter
        self.llm = llm

    def __call__(self, prompt, prompt_constraints=[]):
        """Augments the given prompt using the augmenter and generates
        responses using the LLM.

        Args:
            prompt (:obj:`str`): the prompt to augment and generate responses
            prompt_constraints (List(textattack.constraints.PreTransformationConstraint)): a list of pretransformation
                constraints to apply to the given prompt

        Returns a list of tuples of strings, where the first string in the pair is the augmented prompt and the second
        is the response to the augmented prompt from the LLM
        """
        for constraint in prompt_constraints:
            if isinstance(constraint, PreTransformationConstraint):
                self.augmenter.pre_transformation_constraints.append(constraint)
            else:
                raise ValueError(
                    "Prompt constraints must be of type PreTransformationConstraint"
                )

        augmented_prompts = self.augmenter.augment(prompt)
        for _ in range(len(prompt_constraints)):
            self.augmenter.pre_transformation_constraints.pop()

        outputs = []
        for augmented_prompt in augmented_prompts:
            outputs.append((augmented_prompt, self.llm(augmented_prompt)))
        return outputs
