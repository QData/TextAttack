"""
Model Wrappers
--------------------------
TextAttack can attack any model that takes a list of strings as input and outputs a list of predictions. This is the idea behind *model wrappers*: to help your model conform to this API, we've provided the ``textattack.models.wrappers.ModelWrapper`` abstract class.


We've also provided implementations of model wrappers for common patterns in some popular machine learning frameworks:

"""

from .model_wrapper import ModelWrapper

from .huggingface_model_wrapper import HuggingFaceModelWrapper
from .pytorch_model_wrapper import PyTorchModelWrapper
from .sklearn_model_wrapper import SklearnModelWrapper
from .tensorflow_model_wrapper import TensorFlowModelWrapper
