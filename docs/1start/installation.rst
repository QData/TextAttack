
.. _installation:


Installation
==============

To use TextAttack, you must be running Python 3.6+. Tensorflow needs to be installed for users, and Java needs to be installed for developers. A CUDA-compatible GPU is optional but will greatly improve speed. To install, simply run::

    pip install textattack 

You're now all set to use TextAttack! Try running an attack from the command line::

    textattack attack --recipe textfooler --model bert-base-uncased-mr --num-examples 10

This will run an attack using the TextFooler_ recipe, attacking BERT fine-tuned on the MR dataset. It will attack the first 10 samples. Once everything downloads and starts running, you should see attack results print to ``stdout``.

Read on for more information on TextAttack, including how to use it from a Python script (``import textattack``).

.. _TextFooler: https://arxiv.org/abs/1907.11932
