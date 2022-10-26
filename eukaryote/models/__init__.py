""".. _models:

Models
=========

TextAttack can attack any model that takes a list of strings as input and outputs a list of predictions. This is the idea behind *model wrappers*: to help your model conform to this API, we've provided the ``textattack.models.wrappers.ModelWrapper`` abstract class.

We've also provided implementations of model wrappers for common patterns in some popular machine learning frameworks:


Models User-specified
--------------------------

TextAttack allows users to provide their own models for testing. Models can be loaded in three ways:

1. ``--model`` for pre-trained models and models trained with TextAttack
2. ``--model-from-huggingface`` which will attempt to load any model from the ``HuggingFace model hub <https://huggingface.co/models>``
3. ``--model-from-file`` which will dynamically load a Python file and look for the ``model`` variable



Models Pre-trained
--------------------------

TextAttack also provides lots of pre-trained models for common tasks. Testing different attacks on the same model ensures attack comparisons are fair.

Any of these models can be provided to ``textattack attack`` via ``--model``, for example, ``--model bert-base-uncased-mr``. For a full list of pre-trained models, see the `pre-trained models README <https://github.com/QData/TextAttack/tree/master/textattack/models>`_.


Model Wrappers
--------------------------
TextAttack can attack any model that takes a list of strings as input and outputs a list of predictions. This is the idea behind *model wrappers*: to help your model conform to this API, we've provided the ``textattack.models.wrappers.ModelWrapper`` abstract class.


We've also provided implementations of model wrappers for common patterns in some popular machine learning frameworks: including pytorch / sklearn / tensorflow.
"""


from . import helpers
from . import tokenizers
from . import wrappers
