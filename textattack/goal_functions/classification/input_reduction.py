from .classification_goal_function import ClassificationGoalFunction


class InputReduction(ClassificationGoalFunction):
    """
    An targeted attack on classification models which attempts to maximize the 
    score of the target label until it is the predicted label.
    """

    def __init__(self, *args, target_num_words=1, **kwargs):
        self.target_num_words = target_num_words
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, attacked_text):
        return (
            self.ground_truth_output == model_output.argmax()
            and attacked_text.num_words <= target_num_words
        )

    def _should_skip(self, model_output, attacked_text):
        print(f'gt: {self.ground_truth_output}')
        return self.ground_truth_output != model_output.argmax()

    def _get_score(self, model_output, attacked_text):
        if self.ground_truth_output != model_output.argmax():
            return float("-inf")
        cur_num_words = attacked_text.num_words
        initial_num_words = self.initial_attacked_text.num_words
        num_words_score = (initial_num_words - cur_num_words) / initial_num_words
        model_score = model_output[self.ground_truth_output]
        return num_words_score + model_score / initial_num_words

    def _get_displayed_output(self, raw_output):
        return int(raw_output.argmax())

    def extra_repr_keys(self):
        return ["target_num_words"]
