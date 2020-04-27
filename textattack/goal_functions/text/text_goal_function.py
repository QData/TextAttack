from textattack.goal_functions import GoalFunction

class TextGoalFunction(GoalFunction):
    """"" A goal function defined on a model that outputs text.
        
        model: The PyTorch or TensorFlow model used for evaluation.
        original_output: the original output of the model
    """
    def __init__(self, model):
        super().__init__(model)
        
    def _process_model_outputs(self, _, outputs):
        """ Processes and validates a list of model outputs. 
        
            This is a task-dependent operation. For example, classification 
            outputs need to have a softmax applied. 
        """
        return outputs
        
    def _get_displayed_output(self, raw_output):
        return raw_output