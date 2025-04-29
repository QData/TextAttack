from .classification_goal_function import ClassificationGoalFunction

class LogitSum(ClassificationGoalFunction):
    """

    """

    def __init__(self, *args, target_logit_sum=None, first_element_threshold=None, **kwargs):
        if kwargs.get("validate_outputs", False) is True:
            raise ValueError("LogitSum must be created with validate_outputs=False.")
        
        if ((target_logit_sum is not None) and (first_element_threshold is not None)):
            raise ValueError("Cannot set both target_logit_sum to True and first_element_threshold!")

        self.target_logit_sum = target_logit_sum
        
        if (target_logit_sum is not None) or (first_element_threshold is not None):
            self.first_element_threshold = first_element_threshold
        else:
            self.first_element_threshold = 0.5 # default

        super().__init__(*args, validate_outputs=False, **kwargs)

    def _is_goal_complete(self, model_output, attacked_text):

        if self.target_logit_sum is not None:
            return sum(model_output) < self.target_logit_sum

        return model_output[0] < self.first_element_threshold

    def _get_score(self, model_output, _):
        """
        model_output is a tensor of logits, one for each label.
        """
        return -sum(model_output)
