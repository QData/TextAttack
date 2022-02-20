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


.. code-block::

    >>> dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")
    >>> attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    >>> # Attack 20 samples with CSV logging and checkpoint saved every 5 interval
    >>> attack_args = textattack.AttackArgs(num_examples=20, log_to_csv="log.csv", checkpoint_interval=5, checkpoint_dir="checkpoints", disable_stdout=True)
    >>> attacker = textattack.Attacker(attack, dataset, attack_args)
    >>> attacker.attack_dataset()


.. image:: ../_static/imgs/overview.png
