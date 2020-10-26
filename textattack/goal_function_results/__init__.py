"""
Goal Function Result:
========================

Goal function results report the result of a goal function evaluation, indicating whether an attack succeeded for a given example.

"""
from .goal_function_result import GoalFunctionResult, GoalFunctionResultStatus

from .classification_goal_function_result import ClassificationGoalFunctionResult
from .text_to_text_goal_function_result import TextToTextGoalFunctionResult
