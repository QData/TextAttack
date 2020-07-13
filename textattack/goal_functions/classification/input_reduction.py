from .classification_goal_function import ClassificationGoalFunction


class InputReduction(ClassificationGoalFunction):
    """Attempts to reduce the input down to as few words as possible while
    maintaining the same predicted label.

    From Feng, Wallace, Grissom, Iyyer, Rodriguez, Boyd-Graber. (2018).
    Pathologies of Neural Models Make Interpretations Difficult. ArXiv,
    abs/1804.07781.
    """

    def __init__(self, *args, target_num_words=1, **kwargs):
        self.target_num_words = target_num_words
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, attacked_text):
        return (
            self.ground_truth_output == model_output.argmax()
            and attacked_text.num_words <= self.target_num_words
        )

    def _should_skip(self, model_output, attacked_text):
        return self.ground_truth_output != model_output.argmax()

    def _get_score(self, model_output, attacked_text):
        # Give the lowest score possible to inputs which don't maintain the ground truth label.
        if self.ground_truth_output != model_output.argmax():
            return 0

        cur_num_words = attacked_text.num_words
        initial_num_words = self.initial_attacked_text.num_words

        # The main goal is to reduce the number of words (num_words_score)
        # Higher model score for the ground truth label is used as a tiebreaker (model_score)
        num_words_score = max(
            (initial_num_words - cur_num_words) / initial_num_words, 0
        )
        model_score = model_output[self.ground_truth_output]
        return min(num_words_score + model_score / initial_num_words, 1)

    def extra_repr_keys(self):
        if self.maximizable:
            return ["maximizable"]
        else:
            return ["maximizable", "target_num_words"]
