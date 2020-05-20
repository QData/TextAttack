Models
===============

TextAttack provides different pre-trained models for testing NLP attacks.

We split models up into two broad categories:

- **Classification**: models that output probability scores for some number of classes. These include models for sentiment classification, topic classification, and entailment.
- **Text-to-text**: models that output a sequence of text. These include models that do translation and summarization.


**Classification models:**

   :ref:`BERT`: ``bert-base-uncased`` fine-tuned on various datasets using transformers_.

   :ref:`LSTM`: a standard LSTM fine-tuned on various datasets.
   
   :ref:`CNN`: a word-CNN fine-tuned on various datasets.


**Text-to-text models:**

   :ref:`T5`: ``T5`` fine-tuned on various datasets using transformers_.
   
   

BERT
********
.. _BERT:

.. automodule:: textattack.models.helpers.bert_for_classification
   :members:


We provide pre-trained BERT models on the following datasets:

.. automodule:: textattack.models.classification.bert.bert_for_ag_news_classification
   :members:

.. automodule:: textattack.models.classification.bert.bert_for_imdb_sentiment_classification
   :members:

.. automodule:: textattack.models.classification.bert.bert_for_mr_sentiment_classification
   :members:

.. automodule:: textattack.models.classification.bert.bert_for_yelp_sentiment_classification
   :members:

.. automodule:: textattack.models.entailment.bert.bert_for_mnli
   :members:

.. automodule:: textattack.models.entailment.bert.bert_for_snli
   :members:
   
LSTM
*******
.. _LSTM:

.. automodule:: textattack.models.helpers.lstm_for_classification
   :members:


We provide pre-trained LSTM models on the following datasets:

.. automodule:: textattack.models.classification.lstm.lstm_for_ag_news_classification
   :members:

.. automodule:: textattack.models.classification.lstm.lstm_for_imdb_sentiment_classification
   :members:

.. automodule:: textattack.models.classification.lstm.lstm_for_mr_sentiment_classification
   :members:

.. automodule:: textattack.models.classification.lstm.lstm_for_yelp_sentiment_classification
   :members:



word-CNN
************
.. _CNN:

.. automodule:: textattack.models.helpers.word_cnn_for_classification
   :members:


We provide pre-trained CNN models on the following datasets:

.. automodule:: textattack.models.classification.cnn.word_cnn_for_ag_news_classification
   :members:

.. automodule:: textattack.models.classification.cnn.word_cnn_for_imdb_sentiment_classification
   :members:

.. automodule:: textattack.models.classification.cnn.word_cnn_for_mr_sentiment_classification
   :members:

.. automodule:: textattack.models.classification.cnn.word_cnn_for_yelp_sentiment_classification
   :members:


.. _T5:

T5
*****************

.. automodule:: textattack.models.helpers.t5_for_text_to_text
   :members:


We provide pre-trained T5 models on the following tasks & datasets:

Translation
##############

.. automodule:: textattack.models.translation.t5.t5_models
   :members:
   
Summarization
##############

.. automodule:: textattack.models.summarization.t5_summarization
   :members:


.. _transformers: https://github.com/huggingface/transformers