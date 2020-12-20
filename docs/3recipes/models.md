# TextAttack Model Zoo

TextAttack is model-agnostic - meaning it can run attacks on models implemented in any deep learning framework. Model objects must be able to take a string (or list of strings) and return an output that can be processed by the goal function. For example, machine translation models take a list of strings as input and produce a list of strings as output. Classification and entailment models return an array of scores. As long as the user's model meets this specification, the model is fit to use with TextAttack.



To help users, TextAttack includes pre-trained models for different common NLP tasks. This makes it easier for
users to get started with TextAttack. It also enables a more fair comparison of attacks from
the literature.


## Available Models

### TextAttack Models
TextAttack has two build-in model types, a 1-layer bidirectional LSTM with a hidden
state size of 150 (`lstm`), and a WordCNN with 3 window sizes
(3, 4, 5) and 100 filters for the window size (`cnn`). Both models set dropout
to 0.3 and use a base of the 200-dimensional GLoVE embeddings.

### `transformers` Models
Along with the `lstm` and `cnn`, you can theoretically fine-tune any model based
in the huggingface [transformers](https://github.com/huggingface/transformers/)
repo. Just type the model name (like `bert-base-cased`) and it will be automatically
loaded.

Here are some models from transformers that have worked well for us:
- `bert-base-uncased` and `bert-base-cased`
- `distilbert-base-uncased` and `distilbert-base-cased`
- `albert-base-v2`
- `roberta-base`
- `xlnet-base-cased`


## Evaluation Results of Available Models

All evaluation results were obtained using `textattack eval` to evaluate models on their default
test dataset (test set, if labels are available, otherwise, eval/validation set). You can use
this command to verify the accuracies for yourself: for example, `textattack eval --model roberta-base-mr`.


The LSTM and wordCNN models' code is available in `textattack.models.helpers`. All other models are transformers
imported from the [`transformers`](https://github.com/huggingface/transformers/) package. To list evaluate all 
TextAttack pretrained models, invoke `textattack eval` without specifying a model: `textattack eval --num-examples 1000`.
All evaluations shown are on the full validation or test set up to 1000 examples.


### `LSTM`

<section>

- AG News (`lstm-ag-news`)
    - `datasets` dataset `ag_news`, split `test`
        - Correct/Whole: 914/1000
        - Accuracy: 91.4%
- IMDB (`lstm-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - Correct/Whole: 883/1000
        - Accuracy: 88.30%
- Movie Reviews [Rotten Tomatoes] (`lstm-mr`)
    - `datasets` dataset `rotten_tomatoes`, split `validation`
        - Correct/Whole: 807/1000 
        - Accuracy: 80.70%
    - `datasets` dataset `rotten_tomatoes`, split `test`
        - Correct/Whole: 781/1000
        - Accuracy: 78.10%
- SST-2 (`lstm-sst2`)
    - `datasets` dataset `glue`, subset `sst2`, split `validation`
        - Correct/Whole: 737/872 
        - Accuracy: 84.52%
- Yelp Polarity (`lstm-yelp`)
    - `datasets` dataset `yelp_polarity`, split `test`
        - Correct/Whole: 922/1000
        - Accuracy: 92.20%

</section> 

### `wordCNN`

<section>


- AG News (`cnn-ag-news`)
    - `datasets` dataset `ag_news`, split `test`
        - Correct/Whole: 910/1000
        - Accuracy: 91.00%
- IMDB (`cnn-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - Correct/Whole: 863/1000
        - Accuracy: 86.30%
- Movie Reviews [Rotten Tomatoes] (`cnn-mr`)
    - `datasets` dataset `rotten_tomatoes`, split `validation`
        - Correct/Whole: 794/1000 
        - Accuracy: 79.40%
    - `datasets` dataset `rotten_tomatoes`, split `test`
        - Correct/Whole: 768/1000
        - Accuracy: 76.80%
- SST-2 (`cnn-sst2`)
    - `datasets` dataset `glue`, subset `sst2`, split `validation`
        - Correct/Whole: 721/872 
        - Accuracy: 82.68%
- Yelp Polarity (`cnn-yelp`)
    - `datasets` dataset `yelp_polarity`, split `test`
        - Correct/Whole: 913/1000
        - Accuracy: 91.30%

</section>


### `albert-base-v2`

<section>

- AG News (`albert-base-v2-ag-news`)
    - `datasets` dataset `ag_news`, split `test`
        - Correct/Whole: 943/1000
        - Accuracy: 94.30%
- CoLA (`albert-base-v2-cola`)
    - `datasets` dataset `glue`, subset `cola`, split `validation`
        - Correct/Whole: 829/1000
        - Accuracy: 82.90%
- IMDB (`albert-base-v2-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - Correct/Whole: 913/1000
        - Accuracy: 91.30%
- Movie Reviews [Rotten Tomatoes] (`albert-base-v2-mr`)
    - `datasets` dataset `rotten_tomatoes`, split `validation`
        - Correct/Whole: 882/1000
        - Accuracy: 88.20%
    - `datasets` dataset `rotten_tomatoes`, split `test`
        - Correct/Whole: 851/1000
        - Accuracy: 85.10%
- Quora Question Pairs (`albert-base-v2-qqp`)
    - `datasets` dataset `glue`, subset `qqp`, split `validation`
        - Correct/Whole: 914/1000
        - Accuracy: 91.40%
- Recognizing Textual Entailment (`albert-base-v2-rte`)
    - `datasets` dataset `glue`, subset `rte`, split `validation`
        - Correct/Whole: 211/277 
        - Accuracy: 76.17%
- SNLI (`albert-base-v2-snli`)
    - `datasets` dataset `snli`, split `test`
        - Correct/Whole: 883/1000
        - Accuracy: 88.30%
- SST-2 (`albert-base-v2-sst2`)
    - `datasets` dataset `glue`, subset `sst2`, split `validation`
        - Correct/Whole: 807/872
        - Accuracy: 92.55%)
- STS-b (`albert-base-v2-stsb`)
    - `datasets` dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.9041359738552746
    - Spearman correlation: 0.8995912861209745
- WNLI (`albert-base-v2-wnli`)
    - `datasets` dataset `glue`, subset `wnli`, split `validation`
        - Correct/Whole: 42/71
        - Accuracy: 59.15%
- Yelp Polarity (`albert-base-v2-yelp`)
    - `datasets` dataset `yelp_polarity`, split `test`
        - Correct/Whole: 963/1000
        - Accuracy: 96.30%

</section>

### `bert-base-uncased`

<section>

- AG News (`bert-base-uncased-ag-news`)
    - `datasets` dataset `ag_news`, split `test`
        - Correct/Whole: 942/1000
        - Accuracy: 94.20%
- CoLA (`bert-base-uncased-cola`)
    - `datasets` dataset `glue`, subset `cola`, split `validation`
        - Correct/Whole: 812/1000
        - Accuracy: 81.20%
- IMDB (`bert-base-uncased-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - Correct/Whole: 919/1000
        - Accuracy: 91.90%
- MNLI matched (`bert-base-uncased-mnli`)
    - `datasets` dataset `glue`, subset `mnli`, split `validation_matched`
        - Correct/Whole: 840/1000
        - Accuracy: 84.00%
- Movie Reviews [Rotten Tomatoes] (`bert-base-uncased-mr`)
    - `datasets` dataset `rotten_tomatoes`, split `validation`
        - Correct/Whole: 876/1000
        - Accuracy: 87.60%
    - `datasets` dataset `rotten_tomatoes`, split `test`
        - Correct/Whole: 838/1000
        - Accuracy: 83.80%
- MRPC (`bert-base-uncased-mrpc`)
    - `datasets` dataset `glue`, subset `mrpc`, split `validation`
        - Correct/Whole: 358/408
        - Accuracy: 87.75%
- QNLI (`bert-base-uncased-qnli`)
    - `datasets` dataset `glue`, subset `qnli`, split `validation`
        - Correct/Whole: 904/1000
        - Accuracy: 90.40%
- Quora Question Pairs (`bert-base-uncased-qqp`)
    - `datasets` dataset `glue`, subset `qqp`, split `validation`
        - Correct/Whole: 924/1000
        - Accuracy: 92.40%
- Recognizing Textual Entailment (`bert-base-uncased-rte`)
    - `datasets` dataset `glue`, subset `rte`, split `validation`
        - Correct/Whole: 201/277 
        - Accuracy: 72.56%
- SNLI (`bert-base-uncased-snli`)
    - `datasets` dataset `snli`, split `test`
        - Correct/Whole: 894/1000
        - Accuracy: 89.40%
- SST-2 (`bert-base-uncased-sst2`)
    - `datasets` dataset `glue`, subset `sst2`, split `validation`
        - Correct/Whole: 806/872
        - Accuracy: 92.43%)
- STS-b (`bert-base-uncased-stsb`)
    - `datasets` dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.8775458937815515
    - Spearman correlation: 0.8773251339980935
- WNLI (`bert-base-uncased-wnli`)
    - `datasets` dataset `glue`, subset `wnli`, split `validation`
        - Correct/Whole: 40/71
        - Accuracy: 56.34%
- Yelp Polarity (`bert-base-uncased-yelp`)
    - `datasets` dataset `yelp_polarity`, split `test`
        - Correct/Whole: 963/1000
        - Accuracy: 96.30%

</section>

### `distilbert-base-cased`

<section>


- CoLA (`distilbert-base-cased-cola`)
    - `datasets` dataset `glue`, subset `cola`, split `validation`
        - Correct/Whole: 786/1000
        - Accuracy: 78.60%
- MRPC (`distilbert-base-cased-mrpc`)
    - `datasets` dataset `glue`, subset `mrpc`, split `validation`
        - Correct/Whole: 320/408
        - Accuracy: 78.43%
- Quora Question Pairs (`distilbert-base-cased-qqp`)
    - `datasets` dataset `glue`, subset `qqp`, split `validation`
        - Correct/Whole: 908/1000
        - Accuracy: 90.80%
- SNLI (`distilbert-base-cased-snli`)
    - `datasets` dataset `snli`, split `test`
        - Correct/Whole: 861/1000
        - Accuracy: 86.10%
- SST-2 (`distilbert-base-cased-sst2`)
    - `datasets` dataset `glue`, subset `sst2`, split `validation`
        - Correct/Whole: 785/872
        - Accuracy: 90.02%)
- STS-b (`distilbert-base-cased-stsb`)
    - `datasets` dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.8421540899520146
    - Spearman correlation: 0.8407155030382939

</section>

### `distilbert-base-uncased`

<section>

- AG News (`distilbert-base-uncased-ag-news`)
    - `datasets` dataset `ag_news`, split `test`
        - Correct/Whole: 944/1000
        - Accuracy: 94.40%
- CoLA (`distilbert-base-uncased-cola`)
    - `datasets` dataset `glue`, subset `cola`, split `validation`
        - Correct/Whole: 786/1000
        - Accuracy: 78.60%
- IMDB (`distilbert-base-uncased-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - Correct/Whole: 903/1000
        - Accuracy: 90.30%
- MNLI matched (`distilbert-base-uncased-mnli`)
    - `datasets` dataset `glue`, subset `mnli`, split `validation_matched`
        - Correct/Whole: 817/1000
        - Accuracy: 81.70%
- MRPC (`distilbert-base-uncased-mrpc`)
    - `datasets` dataset `glue`, subset `mrpc`, split `validation`
        - Correct/Whole: 350/408
        - Accuracy: 85.78%
- QNLI (`distilbert-base-uncased-qnli`)
    - `datasets` dataset `glue`, subset `qnli`, split `validation`
        - Correct/Whole: 860/1000
        - Accuracy: 86.00%
- Recognizing Textual Entailment (`distilbert-base-uncased-rte`)
    - `datasets` dataset `glue`, subset `rte`, split `validation`
        - Correct/Whole: 180/277 
        - Accuracy: 64.98%
- STS-b (`distilbert-base-uncased-stsb`)
    - `datasets` dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.8421540899520146
    - Spearman correlation: 0.8407155030382939
- WNLI (`distilbert-base-uncased-wnli`)
    - `datasets` dataset `glue`, subset `wnli`, split `validation`
        - Correct/Whole: 40/71
        - Accuracy: 56.34%

</section>

### `roberta-base`

<section>

- AG News (`roberta-base-ag-news`)
    - `datasets` dataset `ag_news`, split `test`
        - Correct/Whole: 947/1000
        - Accuracy: 94.70%
- CoLA (`roberta-base-cola`)
    - `datasets` dataset `glue`, subset `cola`, split `validation`
        - Correct/Whole: 857/1000
        - Accuracy: 85.70%
- IMDB (`roberta-base-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - Correct/Whole: 941/1000
        - Accuracy: 94.10%
- Movie Reviews [Rotten Tomatoes] (`roberta-base-mr`)
    - `datasets` dataset `rotten_tomatoes`, split `validation`
        - Correct/Whole: 899/1000
        - Accuracy: 89.90%
    - `datasets` dataset `rotten_tomatoes`, split `test`
        - Correct/Whole: 883/1000
        - Accuracy: 88.30%
- MRPC (`roberta-base-mrpc`)
    - `datasets` dataset `glue`, subset `mrpc`, split `validation`
        - Correct/Whole: 371/408
        - Accuracy: 91.18%
- QNLI (`roberta-base-qnli`)
    - `datasets` dataset `glue`, subset `qnli`, split `validation`
        - Correct/Whole: 917/1000
        - Accuracy: 91.70%
- Recognizing Textual Entailment (`roberta-base-rte`)
    - `datasets` dataset `glue`, subset `rte`, split `validation`
        - Correct/Whole: 217/277 
        - Accuracy: 78.34%
- SST-2 (`roberta-base-sst2`)
    - `datasets` dataset `glue`, subset `sst2`, split `validation`
        - Correct/Whole: 820/872
        - Accuracy: 94.04%)
- STS-b (`roberta-base-stsb`)
    - `datasets` dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.906067852162708
    - Spearman correlation: 0.9025045272903051
- WNLI (`roberta-base-wnli`)
    - `datasets` dataset `glue`, subset `wnli`, split `validation`
        - Correct/Whole: 40/71
        - Accuracy: 56.34%

</section>

### `xlnet-base-cased`

<section>

- CoLA (`xlnet-base-cased-cola`)
    - `datasets` dataset `glue`, subset `cola`, split `validation`
        - Correct/Whole: 800/1000
        - Accuracy: 80.00%
- IMDB (`xlnet-base-cased-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - Correct/Whole: 957/1000
        - Accuracy: 95.70%
- Movie Reviews [Rotten Tomatoes] (`xlnet-base-cased-mr`)
    - `datasets` dataset `rotten_tomatoes`, split `validation`
        - Correct/Whole: 908/1000
        - Accuracy: 90.80%
    - `datasets` dataset `rotten_tomatoes`, split `test`
        - Correct/Whole: 876/1000
        - Accuracy: 87.60%
- MRPC (`xlnet-base-cased-mrpc`)
    - `datasets` dataset `glue`, subset `mrpc`, split `validation`
        - Correct/Whole: 363/408
        - Accuracy: 88.97%
- Recognizing Textual Entailment (`xlnet-base-cased-rte`)
    - `datasets` dataset `glue`, subset `rte`, split `validation`
        - Correct/Whole: 196/277 
        - Accuracy: 70.76%
- STS-b (`xlnet-base-cased-stsb`)
    - `datasets` dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.883111673280641
    - Spearman correlation: 0.8773439961182335
- WNLI (`xlnet-base-cased-wnli`)
    - `datasets` dataset `glue`, subset `wnli`, split `validation`
        - Correct/Whole: 41/71
        - Accuracy: 57.75%

</section>


## How we have trained the TextAttack Models


- By Oct 2020, TextAttack provides users with 82 pre-trained TextAttack models, including word-level LSTM, word-level CNN, BERT, and other transformer based models pre-trained on various datasets provided by [HuggingFace](https://github.com/huggingface/nlp/). 

- Since TextAttack is integrated with the  [https://github.com/huggingface/nlp/](https://github.com/huggingface/nlp) library, it can automatically load the test or validation data set for the corresponding pre-trained model. While the literature has mainly focused on classification and entailment, TextAttack's pretrained models enable research on the robustness of models across all GLUE tasks.

- We host all TextAttack Models at huggingface Model Hub: [https://huggingface.co/textattack](https://huggingface.co/textattack)


### Training details for each TextAttack Model 


All of our models have model cards on the HuggingFace model hub. So for now, the easiest way to figure this out is as follows:


- Please Go to our page on the model hub: [https://huggingface.co/textattack](https://huggingface.co/textattack)

- Find the model you're looking for and select its page, for instance: [https://huggingface.co/textattack/roberta-base-imdb](https://huggingface.co/textattack/roberta-base-imdb)

- Scroll down to the end of the page, looking for **model card** section. Here it is the details of the model training for that specific TextAttack model. 

- BTW: For each of our transformers, we selected the best out of a grid search over a bunch of possible hyperparameters. So the model training hyperparemeter actually varies from model to model.



