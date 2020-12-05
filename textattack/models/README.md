# TextAttack Model Zoo

## More details at [https://textattack.readthedocs.io/en/latest/3recipes/models.html](https://textattack.readthedocs.io/en/latest/3recipes/models.html)

TextAttack includes pre-trained models for different common NLP tasks. This makes it easier for
users to get started with TextAttack. It also enables a more fair comparison of attacks from
the literature.

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
        - True Positive/Positive: 914/1000
        - Accuracy: 91.4%
- IMDB (`lstm-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - True Positive/Positive: 883/1000
        - Accuracy: 88.30%
- Movie Reviews [Rotten Tomatoes] (`lstm-mr`)
    - `datasets` dataset `rotten_tomatoes`, split `validation`
        - True Positive/Positive: 807/1000 
        - Accuracy: 80.70%
    - `datasets` dataset `rotten_tomatoes`, split `test`
        - True Positive/Positive: 781/1000
        - Accuracy: 78.10%
- SST-2 (`lstm-sst2`)
    - `datasets` dataset `glue`, subset `sst2`, split `validation`
        - True Positive/Positive: 737/872 
        - Accuracy: 84.52%
- Yelp Polarity (`lstm-yelp`)
    - `datasets` dataset `yelp_polarity`, split `test`
        - True Positive/Positive: 922/1000
        - Accuracy: 92.20%

</section> 

### `wordCNN`

<section>


- AG News (`cnn-ag-news`)
    - `datasets` dataset `ag_news`, split `test`
        - True Positive/Positive: 910/1000
        - Accuracy: 91.00%
- IMDB (`cnn-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - True Positive/Positive: 863/1000
        - Accuracy: 86.30%
- Movie Reviews [Rotten Tomatoes] (`cnn-mr`)
    - `datasets` dataset `rotten_tomatoes`, split `validation`
        - True Positive/Positive: 794/1000 
        - Accuracy: 79.40%
    - `datasets` dataset `rotten_tomatoes`, split `test`
        - True Positive/Positive: 768/1000
        - Accuracy: 76.80%
- SST-2 (`cnn-sst2`)
    - `datasets` dataset `glue`, subset `sst2`, split `validation`
        - True Positive/Positive: 721/872 
        - Accuracy: 82.68%
- Yelp Polarity (`cnn-yelp`)
    - `datasets` dataset `yelp_polarity`, split `test`
        - True Positive/Positive: 913/1000
        - Accuracy: 91.30%

</section>


### `albert-base-v2`

<section>

- AG News (`albert-base-v2-ag-news`)
    - `datasets` dataset `ag_news`, split `test`
        - True Positive/Positive: 943/1000
        - Accuracy: 94.30%
- CoLA (`albert-base-v2-cola`)
    - `datasets` dataset `glue`, subset `cola`, split `validation`
        - True Positive/Positive: 829/1000
        - Accuracy: 82.90%
- IMDB (`albert-base-v2-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - True Positive/Positive: 913/1000
        - Accuracy: 91.30%
- Movie Reviews [Rotten Tomatoes] (`albert-base-v2-mr`)
    - `datasets` dataset `rotten_tomatoes`, split `validation`
        - True Positive/Positive: 882/1000
        - Accuracy: 88.20%
    - `datasets` dataset `rotten_tomatoes`, split `test`
        - True Positive/Positive: 851/1000
        - Accuracy: 85.10%
- Quora Question Pairs (`albert-base-v2-qqp`)
    - `datasets` dataset `glue`, subset `qqp`, split `validation`
        - True Positive/Positive: 914/1000
        - Accuracy: 91.40%
- Recognizing Textual Entailment (`albert-base-v2-rte`)
    - `datasets` dataset `glue`, subset `rte`, split `validation`
        - True Positive/Positive: 211/277 
        - Accuracy: 76.17%
- SNLI (`albert-base-v2-snli`)
    - `datasets` dataset `snli`, split `test`
        - True Positive/Positive: 883/1000
        - Accuracy: 88.30%
- SST-2 (`albert-base-v2-sst2`)
    - `datasets` dataset `glue`, subset `sst2`, split `validation`
        - True Positive/Positive: 807/872
        - Accuracy: 92.55%)
- STS-b (`albert-base-v2-stsb`)
    - `datasets` dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.9041359738552746
    - Spearman correlation: 0.8995912861209745
- WNLI (`albert-base-v2-wnli`)
    - `datasets` dataset `glue`, subset `wnli`, split `validation`
        - True Positive/Positive: 42/71
        - Accuracy: 59.15%
- Yelp Polarity (`albert-base-v2-yelp`)
    - `datasets` dataset `yelp_polarity`, split `test`
        - True Positive/Positive: 963/1000
        - Accuracy: 96.30%

</section>

### `bert-base-uncased`

<section>

- AG News (`bert-base-uncased-ag-news`)
    - `datasets` dataset `ag_news`, split `test`
        - True Positive/Positive: 942/1000
        - Accuracy: 94.20%
- CoLA (`bert-base-uncased-cola`)
    - `datasets` dataset `glue`, subset `cola`, split `validation`
        - True Positive/Positive: 812/1000
        - Accuracy: 81.20%
- IMDB (`bert-base-uncased-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - True Positive/Positive: 919/1000
        - Accuracy: 91.90%
- MNLI matched (`bert-base-uncased-mnli`)
    - `datasets` dataset `glue`, subset `mnli`, split `validation_matched`
        - True Positive/Positive: 840/1000
        - Accuracy: 84.00%
- Movie Reviews [Rotten Tomatoes] (`bert-base-uncased-mr`)
    - `datasets` dataset `rotten_tomatoes`, split `validation`
        - True Positive/Positive: 876/1000
        - Accuracy: 87.60%
    - `datasets` dataset `rotten_tomatoes`, split `test`
        - True Positive/Positive: 838/1000
        - Accuracy: 83.80%
- MRPC (`bert-base-uncased-mrpc`)
    - `datasets` dataset `glue`, subset `mrpc`, split `validation`
        - True Positive/Positive: 358/408
        - Accuracy: 87.75%
- QNLI (`bert-base-uncased-qnli`)
    - `datasets` dataset `glue`, subset `qnli`, split `validation`
        - True Positive/Positive: 904/1000
        - Accuracy: 90.40%
- Quora Question Pairs (`bert-base-uncased-qqp`)
    - `datasets` dataset `glue`, subset `qqp`, split `validation`
        - True Positive/Positive: 924/1000
        - Accuracy: 92.40%
- Recognizing Textual Entailment (`bert-base-uncased-rte`)
    - `datasets` dataset `glue`, subset `rte`, split `validation`
        - True Positive/Positive: 201/277 
        - Accuracy: 72.56%
- SNLI (`bert-base-uncased-snli`)
    - `datasets` dataset `snli`, split `test`
        - True Positive/Positive: 894/1000
        - Accuracy: 89.40%
- SST-2 (`bert-base-uncased-sst2`)
    - `datasets` dataset `glue`, subset `sst2`, split `validation`
        - True Positive/Positive: 806/872
        - Accuracy: 92.43%)
- STS-b (`bert-base-uncased-stsb`)
    - `datasets` dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.8775458937815515
    - Spearman correlation: 0.8773251339980935
- WNLI (`bert-base-uncased-wnli`)
    - `datasets` dataset `glue`, subset `wnli`, split `validation`
        - True Positive/Positive: 40/71
        - Accuracy: 56.34%
- Yelp Polarity (`bert-base-uncased-yelp`)
    - `datasets` dataset `yelp_polarity`, split `test`
        - True Positive/Positive: 963/1000
        - Accuracy: 96.30%

</section>

### `distilbert-base-cased`

<section>


- CoLA (`distilbert-base-cased-cola`)
    - `datasets` dataset `glue`, subset `cola`, split `validation`
        - True Positive/Positive: 786/1000
        - Accuracy: 78.60%
- MRPC (`distilbert-base-cased-mrpc`)
    - `datasets` dataset `glue`, subset `mrpc`, split `validation`
        - True Positive/Positive: 320/408
        - Accuracy: 78.43%
- Quora Question Pairs (`distilbert-base-cased-qqp`)
    - `datasets` dataset `glue`, subset `qqp`, split `validation`
        - True Positive/Positive: 908/1000
        - Accuracy: 90.80%
- SNLI (`distilbert-base-cased-snli`)
    - `datasets` dataset `snli`, split `test`
        - True Positive/Positive: 861/1000
        - Accuracy: 86.10%
- SST-2 (`distilbert-base-cased-sst2`)
    - `datasets` dataset `glue`, subset `sst2`, split `validation`
        - True Positive/Positive: 785/872
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
        - True Positive/Positive: 944/1000
        - Accuracy: 94.40%
- CoLA (`distilbert-base-uncased-cola`)
    - `datasets` dataset `glue`, subset `cola`, split `validation`
        - True Positive/Positive: 786/1000
        - Accuracy: 78.60%
- IMDB (`distilbert-base-uncased-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - True Positive/Positive: 903/1000
        - Accuracy: 90.30%
- MNLI matched (`distilbert-base-uncased-mnli`)
    - `datasets` dataset `glue`, subset `mnli`, split `validation_matched`
        - True Positive/Positive: 817/1000
        - Accuracy: 81.70%
- MRPC (`distilbert-base-uncased-mrpc`)
    - `datasets` dataset `glue`, subset `mrpc`, split `validation`
        - True Positive/Positive: 350/408
        - Accuracy: 85.78%
- QNLI (`distilbert-base-uncased-qnli`)
    - `datasets` dataset `glue`, subset `qnli`, split `validation`
        - True Positive/Positive: 860/1000
        - Accuracy: 86.00%
- Recognizing Textual Entailment (`distilbert-base-uncased-rte`)
    - `datasets` dataset `glue`, subset `rte`, split `validation`
        - True Positive/Positive: 180/277 
        - Accuracy: 64.98%
- STS-b (`distilbert-base-uncased-stsb`)
    - `datasets` dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.8421540899520146
    - Spearman correlation: 0.8407155030382939
- WNLI (`distilbert-base-uncased-wnli`)
    - `datasets` dataset `glue`, subset `wnli`, split `validation`
        - True Positive/Positive: 40/71
        - Accuracy: 56.34%

</section>

### `roberta-base`

<section>

- AG News (`roberta-base-ag-news`)
    - `datasets` dataset `ag_news`, split `test`
        - True Positive/Positive: 947/1000
        - Accuracy: 94.70%
- CoLA (`roberta-base-cola`)
    - `datasets` dataset `glue`, subset `cola`, split `validation`
        - True Positive/Positive: 857/1000
        - Accuracy: 85.70%
- IMDB (`roberta-base-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - True Positive/Positive: 941/1000
        - Accuracy: 94.10%
- Movie Reviews [Rotten Tomatoes] (`roberta-base-mr`)
    - `datasets` dataset `rotten_tomatoes`, split `validation`
        - True Positive/Positive: 899/1000
        - Accuracy: 89.90%
    - `datasets` dataset `rotten_tomatoes`, split `test`
        - True Positive/Positive: 883/1000
        - Accuracy: 88.30%
- MRPC (`roberta-base-mrpc`)
    - `datasets` dataset `glue`, subset `mrpc`, split `validation`
        - True Positive/Positive: 371/408
        - Accuracy: 91.18%
- QNLI (`roberta-base-qnli`)
    - `datasets` dataset `glue`, subset `qnli`, split `validation`
        - True Positive/Positive: 917/1000
        - Accuracy: 91.70%
- Recognizing Textual Entailment (`roberta-base-rte`)
    - `datasets` dataset `glue`, subset `rte`, split `validation`
        - True Positive/Positive: 217/277 
        - Accuracy: 78.34%
- SST-2 (`roberta-base-sst2`)
    - `datasets` dataset `glue`, subset `sst2`, split `validation`
        - True Positive/Positive: 820/872
        - Accuracy: 94.04%)
- STS-b (`roberta-base-stsb`)
    - `datasets` dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.906067852162708
    - Spearman correlation: 0.9025045272903051
- WNLI (`roberta-base-wnli`)
    - `datasets` dataset `glue`, subset `wnli`, split `validation`
        - True Positive/Positive: 40/71
        - Accuracy: 56.34%

</section>

### `xlnet-base-cased`

<section>

- CoLA (`xlnet-base-cased-cola`)
    - `datasets` dataset `glue`, subset `cola`, split `validation`
        - True Positive/Positive: 800/1000
        - Accuracy: 80.00%
- IMDB (`xlnet-base-cased-imdb`)
    - `datasets` dataset `imdb`, split `test`
        - True Positive/Positive: 957/1000
        - Accuracy: 95.70%
- Movie Reviews [Rotten Tomatoes] (`xlnet-base-cased-mr`)
    - `datasets` dataset `rotten_tomatoes`, split `validation`
        - True Positive/Positive: 908/1000
        - Accuracy: 90.80%
    - `datasets` dataset `rotten_tomatoes`, split `test`
        - True Positive/Positive: 876/1000
        - Accuracy: 87.60%
- MRPC (`xlnet-base-cased-mrpc`)
    - `datasets` dataset `glue`, subset `mrpc`, split `validation`
        - True Positive/Positive: 363/408
        - Accuracy: 88.97%
- Recognizing Textual Entailment (`xlnet-base-cased-rte`)
    - `datasets` dataset `glue`, subset `rte`, split `validation`
        - True Positive/Positive: 196/277 
        - Accuracy: 70.76%
- STS-b (`xlnet-base-cased-stsb`)
    - `datasets` dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.883111673280641
    - Spearman correlation: 0.8773439961182335
- WNLI (`xlnet-base-cased-wnli`)
    - `datasets` dataset `glue`, subset `wnli`, split `validation`
        - True Positive/Positive: 41/71
        - Accuracy: 57.75%

</section>

