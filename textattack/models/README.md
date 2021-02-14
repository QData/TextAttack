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


# More details on TextAttack models (details on NLP task, output type, SOTA on paperswithcode; model card on huggingface):

<section>


Fine-tuned Model                         |  NLP Task                                       |  Input type                                   |  Output Type                                        |  paperswithcode.com SOTA                                                       |  huggingface.co Model Card
------------------------------|-----------------------------|------------------------------|-----------------------------|------------------------------|-------------------------------------
albert-base-v2-CoLA                      |  linguistic acceptability                       |  single sentences                             |  binary (1=acceptable/ 0=unacceptable)               |  <sub><sup>https://paperswithcode.com/sota/linguistic-acceptability-on-cola  </sub></sup>            |  <sub><sup>https://huggingface.co/textattack/albert-base-v2-CoLA </sub></sup>
bert-base-uncased-CoLA                   |  linguistic acceptability                       |  single sentences                             |  binary (1=acceptable/ 0=unacceptable)               |  none yet                                                                      |  <sub><sup>https://huggingface.co/textattack/bert-base-uncased-CoLA </sub></sup>
distilbert-base-cased-CoLA               |  linguistic acceptability                       |  single sentences                             |  binary (1=acceptable/ 0=unacceptable)               | <sub><sup> https://paperswithcode.com/sota/linguistic-acceptability-on-cola  </sub></sup>            |  <sub><sup>https://huggingface.co/textattack/distilbert-base-cased-CoLA </sub></sup>
distilbert-base-uncased-CoLA             |  linguistic acceptability                       |  single sentences                             |  binary (1=acceptable/ 0=unacceptable)               | <sub><sup> https://paperswithcode.com/sota/linguistic-acceptability-on-cola     </sub></sup>         |  <sub><sup>https://huggingface.co/textattack/distilbert-base-uncased-CoLA </sub></sup>
roberta-base-CoLA                        |  linguistic acceptability                       |  single sentences                             |  binary (1=acceptable/ 0=unacceptable)               | <sub><sup> https://paperswithcode.com/sota/linguistic-acceptability-on-cola  </sub></sup>            | <sub><sup> https://huggingface.co/textattack/roberta-base-CoLA </sub></sup>
xlnet-base-cased-CoLA                    |  linguistic acceptability                       |  single sentences                             |  binary (1=acceptable/ 0=unacceptable)               | <sub><sup> https://paperswithcode.com/sota/linguistic-acceptability-on-cola   </sub></sup>            |  <sub><sup>https://huggingface.co/textattack/xlnet-base-cased-CoLA </sub></sup> 
albert-base-v2-RTE                       |  natural language inference                     |  sentence pairs (1 premise and 1 hypothesis)  |  binary(0=entailed/1=not entailed)                  | <sub><sup> https://paperswithcode.com/sota/natural-language-inference-on-rte     </sub></sup>         | <sub><sup> https://huggingface.co/textattack/albert-base-v2-RTE </sub></sup> 
albert-base-v2-snli                      |  natural language inference                     |  sentence pairs                               |  accuracy (0=entailment, 1=neutral,2=contradiction)  |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/albert-base-v2-snli </sub></sup>  
albert-base-v2-WNLI                      |  natural language inference                     |  sentence pairs                               |  binary                                             | <sub><sup>  https://paperswithcode.com/sota/natural-language-inference-on-wnli  </sub></sup>           | <sub><sup> https://huggingface.co/textattack/albert-base-v2-WNLI</sub></sup> 
bert-base-uncased-MNLI                   |  natural language inference                     |  sentence pairs (1 premise and 1 hypothesis)  |  accuracy (0=entailment, 1=neutral,2=contradiction)  |  none yet                                                                      |  <sub><sup> https://huggingface.co/textattack/bert-base-uncased-MNLI  </sub></sup>
bert-base-uncased-QNLI                   |  natural language inference                     |  question/answer pairs                        |  binary (1=unanswerable/ 0=answerable)               |  none yet                                                                      |<sub><sup>  https://huggingface.co/textattack/bert-base-uncased-QNLI </sub></sup>
bert-base-uncased-RTE                    |  natural language inference                     |  sentence pairs (1 premise and 1 hypothesis)  |  binary(0=entailed/1=not entailed)                  |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/bert-base-uncased-RTE </sub></sup>
bert-base-uncased-snli                   |  natural language inference                     |  sentence pairs                               |  accuracy (0=entailment, 1=neutral,2=contradiction)  |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/bert-base-uncased-snli </sub></sup>
bert-base-uncased-WNLI                   |  natural language inference                     |  sentence pairs                               |  binary                                             |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/bert-base-uncased-WNLI </sub></sup>
distilbert-base-cased-snli               |  natural language inference                     |  sentence pairs                               |  accuracy (0=entailment, 1=neutral,2=contradiction)  |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/distilbert-base-cased-snli </sub></sup>
distilbert-base-uncased-MNLI             |  natural language inference                     |  sentence pairs (1 premise and 1 hypothesis)  |  accuracy (0=entailment,1=neutral, 2=contradiction)  |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/distilbert-base-uncased-MNLI </sub></sup>
distilbert-base-uncased-RTE              |  natural language inference                     |  sentence pairs (1 premise and 1 hypothesis)  |  binary(0=entailed/1=not entailed)                  | <sub><sup> https://paperswithcode.com/sota/natural-language-inference-on-rte   </sub></sup>          | <sub><sup> https://huggingface.co/textattack/distilbert-base-uncased-RTE</sub></sup>
distilbert-base-uncased-WNLI             |  natural language inference                     |  sentence pairs                               |  binary                                             | <sub><sup> https://paperswithcode.com/sota/natural-language-inference-on-wnli    </sub></sup>        | <sub><sup> https://huggingface.co/textattack/distilbert-base-uncased-WNLI </sub></sup>
roberta-base-QNLI                        |  natural language inference                     |  question/answer pairs                        |  binary (1=unanswerable/ 0=answerable)               | <sub><sup> https://paperswithcode.com/sota/natural-language-inference-on-qnli   </sub></sup>         | <sub><sup>  https://huggingface.co/textattack/roberta-base-QNLI </sub></sup>
roberta-base-RTE                         |  natural language inference                     |  sentence pairs (1 premise and 1 hypothesis)  |  binary(0=entailed/1=not entailed)                  | <sub><sup>  https://paperswithcode.com/sota/natural-language-inference-on-rte  </sub></sup>           | <sub><sup> https://huggingface.co/textattack/roberta-base-RTE</sub></sup>
roberta-base-WNLI                        |  natural language inference                     |  sentence pairs                               |  binary                                             | <sub><sup> https://paperswithcode.com/sota/natural-language-inference-on-wnli </sub></sup>           |  https://huggingface.co/textattack/roberta-base-WNLI </sub></sup>
xlnet-base-cased-RTE                     |  natural language inference                     |  sentence pairs (1 premise and 1 hypothesis)  |  binary(0=entailed/1=not entailed)                  | <sub><sup>  https://paperswithcode.com/sota/ </sub></sup>natural-language-inference-on-rte             | <sub><sup> https://huggingface.co/textattack/xlnet-base-cased-RTE </sub></sup>
xlnet-base-cased-WNLI                    |  natural language inference                     |  sentence pairs                               |  binary                                             |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/xlnet-base-cased-WNLI </sub></sup>
albert-base-v2-QQP                       |  paraphase similarity                           |  question pairs                               |  binary (1=similar/0=not similar)                   | <sub><sup> https://paperswithcode.com/sota/question-answering-on-quora-question-pairs  </sub></sup>  | <sub><sup> https://huggingface.co/textattack/albert-base-v2-QQP</sub></sup>
bert-base-uncased-QQP                    |  paraphase similarity                           |  question pairs                               |  binary (1=similar/0=not similar)                   | <sub><sup> https://paperswithcode.com/sota/question-answering-on-quora-question-pairs  </sub></sup>  | <sub><sup> https://huggingface.co/textattack/bert-base-uncased-QQP </sub></sup>
distilbert-base-uncased-QNLI             |  question answering/natural language inference  |  question/answer pairs                        |  binary (1=unanswerable/ 0=answerable)               | <sub><sup> https://paperswithcode.com/sota/natural-language-inference-on-qnli   </sub></sup>         | <sub><sup> https://huggingface.co/textattack/distilbert-base-uncased-QNLI </sub></sup>
distilbert-base-cased-QQP                |  question answering/paraphase similarity        |  question pairs                               |  binary (1=similar/ 0=not similar)                   | <sub><sup> https://paperswithcode.com/sota/question-answering-on-quora-question-pairs  </sub></sup>  | <sub><sup> https://huggingface.co/textattack/distilbert-base-cased-QQP </sub></sup>
albert-base-v2-STS-B                     |  semantic textual similarity                    |  sentence pairs                               |  similarity (0.0 to 5.0)                            | <sub><sup> https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark </sub></sup> | <sub><sup> https://huggingface.co/textattack/albert-base-v2-STS-B </sub></sup>
bert-base-uncased-MRPC                   |  semantic textual similarity                    |  sentence pairs                               |  binary (1=similar/0=not similar)                   |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/bert-base-uncased-MRPC </sub></sup>
bert-base-uncased-STS-B                  |  semantic textual similarity                    |  sentence pairs                               |  similarity (0.0 to 5.0)                            |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/bert-base-uncased-STS-B </sub></sup>
distilbert-base-cased-MRPC               |  semantic textual similarity                    |  sentence pairs                               |  binary (1=similar/0=not similar)                   | <sub><sup> https://paperswithcode.com/sota/semantic-textual-similarity-on-mrpc  </sub></sup>         | <sub><sup> https://huggingface.co/textattack/distilbert-base-cased-MRPC </sub></sup>
distilbert-base-cased-STS-B              |  semantic textual similarity                    |  sentence pairs                               |  similarity (0.0 to 5.0)                            | <sub><sup> https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark </sub></sup> | <sub><sup> https://huggingface.co/textattack/distilbert-base-cased-STS-B </sub></sup>
distilbert-base-uncased-MRPC             |  semantic textual similarity                    |  sentence pairs                               |  binary (1=similar/0=not similar)                   | <sub><sup> https://paperswithcode.com/sota/semantic-textual-similarity-on-mrpc   </sub></sup>        | <sub><sup> https://huggingface.co/textattack/distilbert-base-uncased-MRPC</sub></sup>
roberta-base-MRPC                        |  semantic textual similarity                    |  sentence pairs                               |  binary (1=similar/0=not similar)                   | <sub><sup> https://paperswithcode.com/sota/semantic-textual-similarity-on-mrpc </sub></sup>          | <sub><sup> https://huggingface.co/textattack/roberta-base-MRPC </sub></sup>
roberta-base-STS-B                       |  semantic textual similarity                    |  sentence pairs                               |  similarity (0.0 to 5.0)                            | <sub><sup> https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark </sub></sup> | <sub><sup> https://huggingface.co/textattack/roberta-base-STS-B </sub></sup>
xlnet-base-cased-MRPC                    |  semantic textual similarity                    |  sentence pairs                               |  binary (1=similar/0=not similar)                   | <sub><sup> https://paperswithcode.com/sota/semantic-textual-similarity-on-mrpc </sub></sup>          | <sub><sup> https://huggingface.co/textattack/xlnet-base-cased-MRPC </sub></sup>
xlnet-base-cased-STS-B                   |  semantic textual similarity                    |  sentence pairs                               |  similarity (0.0 to 5.0)                            | <sub><sup> https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark </sub></sup> | <sub><sup> https://huggingface.co/textattack/xlnet-base-cased-STS-B </sub></sup>
albert-base-v2-imdb                      |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/albert-base-v2-imdb </sub></sup>
albert-base-v2-rotten-tomatoes           |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/albert-base-v2-rotten-tomatoes </sub></sup>
albert-base-v2-SST-2                     |  sentiment analysis                             |  phrases                                      |  accuracy (0.0000 to 1.0000)                        | <sub><sup> https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary  </sub></sup>          | <sub><sup> https://huggingface.co/textattack/albert-base-v2-SST-2 </sub></sup>
albert-base-v2-yelp-polarity             |  sentiment analysis                             |  yelp reviews                                 |  binary (1=good/0=bad)                              |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/albert-base-v2-yelp-polarity </sub></sup>
bert-base-uncased-imdb                   |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/bert-base-uncased-imdb </sub></sup>
bert-base-uncased-rotten-tomatoes        |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/bert-base-uncased-rotten-tomatoes </sub></sup>
bert-base-uncased-SST-2                  |  sentiment analysis                             |  phrases                                      |  accuracy (0.0000 to 1.0000)                        | <sub><sup> https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary </sub></sup>           | <sub><sup> https://huggingface.co/textattack/bert-base-uncased-SST-2 </sub></sup>
bert-base-uncased-yelp-polarity          |  sentiment analysis                             |  yelp reviews                                 |  binary (1=good/0=bad)                              | <sub><sup> https://paperswithcode.com/sota/sentiment-analysis-on-yelp-binary  </sub></sup>           | <sub><sup> https://huggingface.co/textattack/bert-base-uncased-yelp-polarity </sub></sup>
cnn-imdb                                 |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              | <sub><sup> https://paperswithcode.com/sota/sentiment-analysis-on-imdb  </sub></sup>                  |  none
cnn-mr                                   |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              |  none yet                                                                      |  none
cnn-sst2                                 |  sentiment analysis                             |  phrases                                      |  accuracy (0.0000 to 1.0000)                        |  <sub><sup> https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary </sub></sup>           |  none
cnn-yelp                                 |  sentiment analysis                             |  yelp reviews                                 |  binary (1=good/0=bad)                              | <sub><sup> https://paperswithcode.com/sota/sentiment-analysis-on-yelp-binary </sub></sup>            |  none
distilbert-base-cased-SST-2              |  sentiment analysis                             |  phrases                                      |  accuracy (0.0000 to 1.0000)                        | <sub><sup> https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary </sub></sup>           | <sub><sup> https://huggingface.co/textattack/distilbert-base-cased-SST-2 </sub></sup>
distilbert-base-uncased-imdb             |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              | <sub><sup> https://paperswithcode.com/sota/sentiment-analysis-on-imdb</sub></sup>                    | <sub><sup> https://huggingface.co/textattack/distilbert-base-uncased-imdb </sub></sup>
distilbert-base-uncased-rotten-tomatoes  |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/distilbert-base-uncased-rotten-tomatoes </sub></sup>
lstm-imdb                                |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              | <sub><sup> https://paperswithcode.com/sota/sentiment-analysis-on-imdb </sub></sup>                    |  none
lstm-mr                                  |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              |  none yet                                                                      |  none
lstm-sst2                                |  sentiment analysis                             |  phrases                                      |  accuracy (0.0000 to 1.0000)                        |  none yet                                                                      |  none
lstm-yelp                                |  sentiment analysis                             |  yelp reviews                                 |  binary (1=good/0=bad)                              |  none yet                                                                      |  none
roberta-base-imdb                        |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/roberta-base-imdb </sub></sup>
roberta-base-rotten-tomatoes             |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/roberta-base-rotten-tomatoes </sub></sup>
roberta-base-SST-2                       |  sentiment analysis                             |  phrases                                      |  accuracy (0.0000 to 1.0000)                        | <sub><sup> https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary  </sub></sup>          | <sub><sup> https://huggingface.co/textattack/roberta-base-SST-2 </sub></sup>
xlnet-base-cased-imdb                    |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/xlnet-base-cased-imdb </sub></sup>
xlnet-base-cased-rotten-tomatoes         |  sentiment analysis                             |  movie reviews                                |  binary (1=good/0=bad)                              |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/xlnet-base-cased-rotten-tomatoes </sub></sup>
albert-base-v2-ag-news                   |  text classification                            |  news articles                                |  news category                                      |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/albert-base-v2-ag-news </sub></sup>
bert-base-uncased-ag-news                |  text classification                            |  news articles                                |  news category                                      |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/bert-base-uncased-ag-news </sub></sup>
cnn-ag-news                              |  text classification                            |  news articles                                |  news category                                      | <sub><sup> https://paperswithcode.com/sota/text-classification-on-ag-news  </sub></sup>              |  none
distilbert-base-uncased-ag-news          |  text classification                            |  news articles                                |  news category                                      |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/distilbert-base-uncased-ag-news </sub></sup>
lstm-ag-news                             |  text classification                            |  news articles                                |  news category                                      | <sub><sup> https://paperswithcode.com/sota/text-classification-on-ag-news  </sub></sup>              |  none
roberta-base-ag-news                     |  text classification                            |  news articles                                |  news category                                      |  none yet                                                                      | <sub><sup> https://huggingface.co/textattack/roberta-base-ag-news </sub></sup>

</section>
