from textattack.goal_functions import GoalFunction

class UntargetedClassification(GoalFunction):
    
    def _is_goal_complete(self, model_output, correct_output):
        return model_output.argmax() != correct_output 

    def _get_score(self, model_output, correct_output):
        return -model_output[correct_output]

    def _get_displayed_output(self, raw_output):
        return int(raw_output.argmax())
        
