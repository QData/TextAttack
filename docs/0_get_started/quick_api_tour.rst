Quick Tour
==========================

Let us have a quick look at how TextAttack can be used to carry out adversarial attack.

Attacking a BERT model
------------------------------
Let us attack a BERT model fine-tuned for sentimental classification task. We are going to use a model that has already been fine-tuned on IMDB dataset using the Transformers library. 

.. code-block::

    >>> import transformers
    >>> model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    >>> tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")


TextAttack requires both the model and the tokenizer to be wrapped by a :class:`~transformers.models.wrapper.ModelWrapper` class that implements the forward pass operation given a list of input texts. For models provided by Transformers library, we can also simply use :class:`~transformers.models.wrapper.HuggingFaceModelWrapper` class which implements both the forward pass and tokenization.

.. code-block::

    >>> import textattack
    >>> model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

Next, let's build the attack that we want to use. TextAttack provides prebuilt attacks in the form of :class:`~transformers.attack_recipes.AttackRecipe`. For this example, we will use :ref:TextFooler attack 

Let us also load the IMDB dataset using 🤗 Datasets library. TextAttack also requires that the dataset

.. code-block::

    >>> import datasets
    >>> dataset = datasets.load_dataset("imdb", split="test")


