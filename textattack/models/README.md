# TextAttack Models

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

- IMDB (`lstm-imdb`)
    - nlp dataset `imdb`, split `test`
    - Successes: 883/1000
    - Accuracy: 88.30%
- Movie Reviews [Rotten Tomatoes] (`lstm-mr`)
    - nlp dataset `rotten_tomatoes`, split `test`
    - Successes: 781/1000
    - Accuracy: 78.10%
- SST-2 (`lstm-sst2`)
    - nlp dataset `glue`, subset `sst2`, split `validation`
    - Successes: 737/872 
    - Accuracy: 84.52%
- Yelp Polarity (`lstm-yelp`)
    - nlp dataset `yelp_polarity`, split `test`
    - Successes: 922/1000
    - Accuracy: 92.20%

</section> 

### `wordCNN`

<section>


- IMDB (`cnn-imdb`)
    - nlp dataset `imdb`, split `test`
    - Successes: 863/1000
    - Accuracy: 86.30%
- Movie Reviews [Rotten Tomatoes] (`cnn-mr`)
    - nlp dataset `rotten_tomatoes`, split `test`
    - Successes: 768/1000
    - Accuracy: 76.80%
- SST-2 (`lstm-sst2`)
    - nlp dataset `glue`, subset `sst2`, split `validation`
    - Successes: 721/872 
    - Accuracy: 82.68%
- Yelp Polarity (`lstm-yelp`)
    - nlp dataset `yelp_polarity`, split `test`
    - Successes: 913/1000
    - Accuracy: 91.30%

</section>


### `albert-base-v2`

<section>

- CoLA `albert-base-v2-cola`
    - nlp dataset `glue`, subset `cola`, split `validation`
    - Successes: 829/1000
    - Accuracy: 82.90%
- IMDB (`albert-base-v2-imdb`)
    - nlp dataset `imdb`, split `test`
    - Successes: 913/1000
    - Accuracy: 91.30%
- Movie Reviews [Rotten Tomatoes] (`albert-base-v2-mr`)
    - nlp dataset `rotten_tomatoes`, split `test`
    - Successes: 851/1000
    - Accuracy: 85.10%
- Quora Question Pairs (`albert-base-v2-qqp`)
    - nlp dataset `glue`, subset `qqp`, split `validation`
    - Successes: 914/1000
    - Accuracy: 91.40%
- Recognizing Textual Entailment (`albert-base-v2-rte`)
    - nlp dataset `glue`, subset `rte`, split `validation`
    - Successes: 211/277 
    - Accuracy: 76.17%
- SNLI (`albert-base-v2-snli`)
    - nlp dataset `snli`, split `test`
    - Successes: 883/1000
    - Accuracy: 88.30%
- SST-2 (`albert-base-v2-sst2`)
    - nlp dataset `glue`, subset `sst2`, split `validation`
    - Successes: 807/872
    - Accuracy: 92.55%)
- STS-b (`albert-base-v2-stsb`)
    - nlp dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.9041359738552746
    - Spearman correlation: 0.8995912861209745
- WNLI (`albert-base-v2-wnli`)
    - nlp dataset `glue`, subset `wnli`, split `validation`
    - Successes: 42/71
    - Accuracy: 59.15%
- Yelp Polarity (`lstm-yelp`)
    - nlp dataset `yelp_polarity`, split `test`
    - Successes: 963/1000
    - Accuracy: 96.30%

</section>

### `bert-base-uncased`

<section>

- CoLA (`bert-base-uncased-cola`)
    - nlp dataset `glue`, subset `cola`, split `validation`
    - Successes: 812/1000
    - Accuracy: 81.20%
- IMDB (`bert-base-uncased-imdb`)
    - nlp dataset `imdb`, split `test`
    - Successes: 919/1000
    - Accuracy: 91.90%
- MNLI matched (`bert-base-uncased-mnli`)
    - nlp dataset `glue`, subset `mnli`, split `validation_matched`
    - Successes: 840/1000
    - Accuracy: 84.00%
- Movie Reviews [Rotten Tomatoes] (`bert-base-uncased-mr`)
    - nlp dataset `rotten_tomatoes`, split `test`
    - Successes: 838/1000
    - Accuracy: 83.80%
- MRPC (`bert-base-uncased-mrpc`)
    - nlp dataset `glue`, subset `mrpc`, split `validation`
    - Successes: 358/408
    - Accuracy: 87.75%
- QNLI (`bert-base-uncased-qnli`)
    - nlp dataset `glue`, subset `qnli`, split `validation`
    - Successes: 904/1000
    - Accuracy: 90.40%
- Quora Question Pairs (`bert-base-uncased-qqp`)
    - nlp dataset `glue`, subset `qqp`, split `validation`
    - Successes: 924/1000
    - Accuracy: 92.40%
- Recognizing Textual Entailment (`bert-base-uncased-rte`)
    - nlp dataset `glue`, subset `rte`, split `validation`
    - Successes: 201/277 
    - Accuracy: 72.56%
- SNLI (`bert-base-uncased-snli`)
    - nlp dataset `snli`, split `test`
    - Successes: 894/1000
    - Accuracy: 89.40%
- SST-2 (`bert-base-uncased-sst2`)
    - nlp dataset `glue`, subset `sst2`, split `validation`
    - Successes: 806/872
    - Accuracy: 92.43%)
- STS-b (`bert-base-uncased-stsb`)
    - nlp dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.8775458937815515
    - Spearman correlation: 0.8773251339980935
- WNLI (`bert-base-uncased-wnli`)
    - nlp dataset `glue`, subset `wnli`, split `validation`
    - Successes: 40/71
    - Accuracy: 56.34%
- Yelp Polarity (`bert-base-uncased-yelp`)
    - nlp dataset `yelp_polarity`, split `test`
    - Successes: 963/1000
    - Accuracy: 96.30%

</section>

### `distilbert-base-cased`

<section>

- CoLA (`distilbert-base-cased-cola`)
    - nlp dataset `glue`, subset `cola`, split `validation`
    - Successes: 786/1000
    - Accuracy: 78.60%
- MRPC (`distilbert-base-cased-mrpc`)
    - nlp dataset `glue`, subset `mrpc`, split `validation`
    - Successes: 320/408
    - Accuracy: 78.43%
- Quora Question Pairs (`distilbert-base-cased-qqp`)
    - nlp dataset `glue`, subset `qqp`, split `validation`
    - Successes: 908/1000
    - Accuracy: 90.80%
- SNLI (`distilbert-base-cased-snli`)
    - nlp dataset `snli`, split `test`
    - Successes: 861/1000
    - Accuracy: 86.10%
- SST-2 (`distilbert-base-cased-sst2`)
    - nlp dataset `glue`, subset `sst2`, split `validation`
    - Successes: 785/872
    - Accuracy: 90.02%)
- STS-b (`distilbert-base-cased-stsb`)
    - nlp dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.8421540899520146
    - Spearman correlation: 0.8407155030382939

</section>

### `distilbert-base-uncased`

<section>

- CoLA (`distilbert-base-uncased-cola`)
    - nlp dataset `glue`, subset `cola`, split `validation`
    - Successes: 786/1000
    - Accuracy: 78.60%
- IMDB (`distilbert-base-uncased-imdb`)
    - nlp dataset `imdb`, split `test`
    - Successes: 903/1000
    - Accuracy: 90.30%
- MNLI matched (`distilbert-base-uncased-mnli`)
    - nlp dataset `glue`, subset `mnli`, split `validation_matched`
    - Successes: 817/1000
    - Accuracy: 81.70%
- MRPC (`distilbert-base-uncased-mrpc`)
    - nlp dataset `glue`, subset `mrpc`, split `validation`
    - Successes: 350/408
    - Accuracy: 85.78%
- QNLI (`distilbert-base-uncased-qnli`)
    - nlp dataset `glue`, subset `qnli`, split `validation`
    - Successes: 860/1000
    - Accuracy: 86.00%
- Recognizing Textual Entailment (`distilbert-base-uncased-rte`)
    - nlp dataset `glue`, subset `rte`, split `validation`
    - Successes: 180/277 
    - Accuracy: 64.98%
- STS-b (`distilbert-base-uncased-stsb`)
    - nlp dataset `glue`, subset `stsb`, split `validation`
    - Pearson correlation: 0.8421540899520146
    - Spearman correlation: 0.8407155030382939
- WNLI (`distilbert-base-uncased-wnli`)
    - nlp dataset `glue`, subset `wnli`, split `validation`
    - Successes: 40/71
    - Accuracy: 56.34%

</section>

### `roberta-base`

<section>



</section>

### `xlnet-base-cased`

<section>
</section>

