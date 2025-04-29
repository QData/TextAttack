from .classification_goal_function import ClassificationGoalFunction

class LogitSum(ClassificationGoalFunction):
    """
    A goal function that minimizes the sum of output logits for classification models.

    This can be used for tasks where the objective is to suppress the model's overall confidence,
    or specifically the logit of the most probable label.

    Behavior:
        - If `target_logit_sum` is set, the attack succeeds when the sum of all logits
          is less than `target_logit_sum`.
        - If `first_element_threshold` is set (or defaulted to 0.5), the attack succeeds
          when the first logit's value is less than that threshold.

    Args:
        target_logit_sum (float, optional): A threshold for the total sum of logits.
        first_element_threshold (float, optional): A fallback threshold for the first logit only.

    Note:
        This goal function cannot be instantiated with `validate_outputs=True`.
        Only one of `target_logit_sum` or `first_element_threshold` may be set.
    """

    def __init__(self, *args, target_logit_sum=None, first_element_threshold=None, **kwargs):
        """
        Initializes the LogitSum goal function.

        This goal function is used to reduce the model's overall logit output, either by
        minimizing the sum of all logits or by lowering a specific logit's value.

        Args:
            target_logit_sum (float, optional): If set, the attack is successful when the
                sum of all logits is less than this threshold.
            first_element_threshold (float, optional): If `target_logit_sum` is not set,
                this threshold is used to determine success based on whether the first logit's
                value falls below it. Defaults to 0.5 if not specified.

        Keyword Args:
            validate_outputs (bool): Must be False. This goal function expects raw logits
                and does not support output validation.

        Raises:
            ValueError: If `validate_outputs=True`, or if both `target_logit_sum` and 
            `first_element_threshold` are set at the same time.
        """
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
