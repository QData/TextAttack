Models
===============

TextAttack provides different pre-trained models for testing NLP attacks.

We split models up into two broad categories:

- **Classification**: models that output probability scores for some number of classes. These include models for sentiment classification, topic classification, and entailment.
- **Text-to-text**: models that output a sequence of text. These include models that do translation and summarization.


Classification models
=======================

   :ref:`BERT`: ``bert-base-uncased`` fine-tuned on various datasets using transformers_.

   :ref:`LSTM`: a standard LSTM fine-tuned on various datasets.
   
   :ref:`CNN`: a word-CNN fine-tuned on various datasets.


Text-to-text models
=======================

   :ref:`T5`: ``T5`` fine-tuned on various datasets using transformers_.


.. _transformers: https://github.com/huggingface/transformers