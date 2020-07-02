Models
===============

TextAttack provides different pre-trained models for testing NLP attacks.

We split models up into two broad categories:

- **Classification**: models that output probability scores for some number of classes. These include models for sentiment classification, topic classification, and entailment.
- **Text-to-text**: models that output a sequence of text. These include models that do translation and summarization.


**Classification models:**

   :ref:`BERT`: ``bert-base-uncased`` fine-tuned on various datasets using transformers_.

   :ref:`LSTM`: a standard LSTM fine-tuned on various datasets.
   
   :ref:`CNN`: a Word-CNN fine-tuned on various datasets.


**Text-to-text models:**

   :ref:`T5`: ``T5`` fine-tuned on various datasets using transformers_.
   
   

BERT
********
.. _BERT:

.. automodule:: textattack.models.helpers.bert_for_classification
   :members:


LSTM
*******
.. _LSTM:

.. automodule:: textattack.models.helpers.lstm_for_classification
   :members:


Word-CNN
************
.. _CNN:

.. automodule:: textattack.models.helpers.word_cnn_for_classification
   :members:

.. _T5:

T5
*****************

.. automodule:: textattack.models.helpers.t5_for_text_to_text
   :members:
