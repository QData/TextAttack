class GoalFunctionResult:
    
    def __init__(self, tokenized_text, output, succeeded, score):
        self.tokenized_text = tokenized_text
        self.output = output
        self.score = score
        self.succeeded = succeeded
