class GoalFunctionResult:
    """
    Represents the result of a goal function evaluating a TokenizedText object.
    Args:
        tokenized_text: The sequence that was evaluated.
        output: The display-friendly output.
        succeeded: Whether the goal has been achieved.
        score: A score representing how close the model is to achieving its goal.
    """
    def __init__(self, tokenized_text, output, succeeded, score):
        self.tokenized_text = tokenized_text
        self.output = output
        self.score = score
        self.succeeded = succeeded
