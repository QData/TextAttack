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

|          Model        |           Dataset   		     |   Dataset Descriptor   |   Successes   |    Accuracy   |  Technical Details                                          |
| :-------------------: | :----------------------------: | :--------------------: | :-----------: | :-----------: | :---------------------------------------------------------: |
| `LSTM`                | AG News             			 |  `lstm-ag-news`        | 914/1000      |  91.4%        | `datasets` dataset `ag_news`, split `test`                  |
|                       | IMDB         					 |  `lstm-imdb`           | 883/1000      |  88.30%       | `datasets` dataset `imdb`, split `test`                     |
|         				| Movie Reviews [Rotten Tomatoes]|  `lstm-mr`             | 807/1000      |  80.70%       | `datasets` dataset `rotten_tomatoes`, split `validation`    |
|         				| Movie Reviews [Rotten Tomatoes]|  `lstm-mr`             | 781/1000      |  78.10%       | `datasets` dataset `rotten_tomatoes`, split `test`          |
|         				| SST-2                          |  `lstm-sst2`           | 737/872       |  84.52%       | `datasets` dataset `glue`, subset `sst2`, split `validation`|
|                       | Yelp Polarity                  |  `lstm-yelp`           | 922/1000      |  92.20%       | `datasets` dataset `yelp_polarity`, split `test`            |

### `wordCNN`

|          Model        |           Dataset   		     |   Dataset Descriptor   |   Successes   |    Accuracy   |  Technical Details                                          |
| :-------------------: | :----------------------------: | :--------------------: | :-----------: | :-----------: | :---------------------------------------------------------: |
| `wordCNN`				| AG News                        |  `cnn-ag-news`         | 910/1000      |  91.00%       | `datasets` dataset `ag_news`, split `test`                  |
|                       | IMDB                           |  `cnn-imdb`            | 863/1000      |  86.30%       | `datasets` dataset `imdb`, split `test`                     |
|                       | Movie Reviews [Rotten Tomatoes]|  `cnn-mr`              | 794/1000      |  79.40%       | `datasets` dataset `rotten_tomatoes`, split `validation`    |
|                       | Movie Reviews [Rotten Tomatoes]|  `cnn-mr`              | 768/1000      |  76.80%       | `datasets` dataset `rotten_tomatoes`, split `test`          |
|                       | SST-2                          |  `cnn-sst2`            | 721/872       |  82.68%       | `datasets` dataset `glue`, subset `sst2`, split `validation`|
|                       | Yelp Polarity                  |  `cnn-yelp`            | 913/1000      |  91.30%       | `datasets` dataset `yelp_polarity`, split `test`            |

### `albert-base-v2`

|          Model        |           Dataset   		     |     Dataset Descriptor    |      Successes      |       Accuracy      |  Technical Details                                          |
| :-------------------: | :----------------------------: | :-----------------------: | :-----------------: | :-----------------: | :---------------------------------------------------------: |
| `albert-base-v2`      | AG News                        |  `albert-base-v2-ag-news` |  943/1000           |  94.30%             | `datasets` dataset `ag_news`, split `test`                  |
|                       | CoLA                           |  `albert-base-v2-cola`    |  829/1000           |  82.90%             | `datasets` dataset `glue`, subset `cola`, split `validation`|
|                       | IMDB                           |  `albert-base-v2-imdb`    |  913/1000           |  91.30%             | `datasets` dataset `imdb`, split `test`                     |
|                       | Movie Reviews [Rotten Tomatoes]|  `albert-base-v2-mr`      |  882/1000           |  88.20%             | `datasets` dataset `rotten_tomatoes`, split `validation`    |
|                       | Movie Reviews [Rotten Tomatoes]|  `albert-base-v2-mr`      |  851/1000           |  85.10%             | `datasets` dataset `rotten_tomatoes`, split `test`          |
|                       | Quora Question Pairs           |  `albert-base-v2-qqp`     |  914/1000           |  91.40%             | `datasets` dataset `glue`, subset `qqp`, split `validation` |
|                       | Recognizing Textual Entailment |  `albert-base-v2-rte`     |  211/277            |  76.17%             | `datasets` dataset `glue`, subset `rte`, split `validation` |
|                       | SNLI                           |  `albert-base-v2-snli`    |  883/1000           |  88.30%             | `datasets` dataset `snli`, split `test`                     |
|                       | SST-2                          |  `albert-base-v2-sst2`    |  807/872            |  92.55%             | `datasets` dataset `glue`, subset `sst2`, split `validation`|
|                       | STS-b                          |  `albert-base-v2-stsb`    |  0.9041359738552746 |  0.8995912861209745 | `datasets` dataset `glue`, subset `stsb`, split `validation`|
|                       | WNLI                           |  `albert-base-v2-wnli`    |  42/71              |  59.15%             | `datasets` dataset `glue`, subset `wnli`, split `validation`|
|                       | Yelp Polarity                  |  `albert-base-v2-yelp`    |  963/1000           |  96.30%             | `datasets` dataset `yelp_polarity`, split `test`            |

Footnote: For dataset STS-b, Successes = Pearson correlation, Accuracy = Spearman correlation

### `bert-base-uncased`

|          Model        |           Dataset   		     |       Dataset Descriptor      |      Successes      |       Accuracy      |  Technical Details                                                  |
| :-------------------: | :----------------------------: | :---------------------------: | :-----------------: | :-----------------: | :-----------------------------------------------------------------: |
| `bert-base-uncased`   | AG News                        |  `bert-base-uncased-ag-news`  |  942/1000           |  94.20%             | `datasets` dataset `ag_news`, split `test`                          |
|                       | CoLA                           |  `bert-base-uncased-cola`     |  812/1000           |  81.20%             | `datasets` dataset `glue`, subset `cola`, split `validation`        |
|                       | IMDB                           |  `bert-base-uncased-imdb`     |  919/1000           |  91.90%             | `datasets` dataset `imdb`, split `test`                             |
|                       | MNLI matched                   |  `bert-base-uncased-mnli`     |  840/1000           |  84.00%             | `datasets` dataset `glue`, subset `mnli`, split `validation_matched`|
|                       | Movie Reviews [Rotten Tomatoes]|  `bert-base-uncased-mr`       |  876/1000           |  87.60%             | `datasets` dataset `rotten_tomatoes`, split `validation`            |
|                       | Movie Reviews [Rotten Tomatoes]|  `bert-base-uncased-mr`       |  838/1000           |  83.80%             | `datasets` dataset `rotten_tomatoes`, split `test`                  |
|                       | MRPC                           |  `bert-base-uncased-mrpc`     |  358/408            |  87.75%             | `datasets` dataset `glue`, subset `mrpc`, split `validation`        |
|                       | QNLI                           |  `bert-base-uncased-qnli`     |  904/1000           |  90.40%             | `datasets` dataset `glue`, subset `qnli`, split `validation`        |
|                       | Quora Question Pairs           |  `bert-base-uncased-qqp`      |  924/1000           |  92.40%             | `datasets` dataset `glue`, subset `qqp`, split `validation`         |
|                       | Recognizing Textual Entailment |  `bert-base-uncased-rte`      |  201/277            |  72.56%             | `datasets` dataset `glue`, subset `rte`, split `validation`         |
|                       | SNLI                           |  `bert-base-uncased-snli`     |  894/1000           |  89.40%             | `datasets` dataset `snli`, split `test`                             |
|                       | SST-2                          |  `bert-base-uncased-sst2`     |  806/872            |  92.43%             | `datasets` dataset `glue`, subset `sst2`, split `validation`        |
|                       | STS-b                          |  `bert-base-uncased-stsb`     |  0.8775458937815515 |  0.8773251339980935 | `datasets` dataset `glue`, subset `stsb`, split `validation`        | 
|                       | WNLI                           |  `bert-base-uncased-wnli`     |  40/71              |  56.34%             | `datasets` dataset `glue`, subset `wnli`, split `validation`	       |
|                       | Yelp Polarity                  |  `bert-base-uncased-yelp`     |  963/1000           |  96.30%             | `datasets` dataset `yelp_polarity`, split `test`                    |

Footnote: For dataset STS-b, Successes = Pearson correlation, Accuracy = Spearman correlation

### `distilbert-base-cased`

|          Model          |           Dataset   		   |         Dataset Descriptor         |      Successes      |       Accuracy      |  Technical Details                                          |
| :---------------------: | :----------------------------: | :--------------------------------: | :-----------------: | :-----------------: | :---------------------------------------------------------: |
| `distilbert-base-cased' | CoLA                           |  `distilbert-base-cased-cola`      |  786/1000           |  78.60%             | `datasets` dataset `glue`, subset `cola`, split `validation`|
|                         | MRPC                           |  `distilbert-base-cased-mrpc`      |  320/408            |  78.43%             | `datasets` dataset `glue`, subset `mrpc`, split `validation`|
|                         | Quora Question Pairs           |  `distilbert-base-cased-qqp`       |  908/1000           |  90.80%             | `datasets` dataset `glue`, subset `qqp`, split `validation` |
|                         | SNLI                           |  `distilbert-base-cased-snli`      |  861/1000           |  86.10%             | `datasets` dataset `snli`, split `test`                     |
|                         | SST-2                          |  `distilbert-base-cased-sst2`      |  785/872            |  90.02%             | `datasets` dataset `glue`, subset `sst2`, split `validation`|
|                         | STS-b                          |  `distilbert-base-cased-stsb`      |  0.8421540899520146 |  0.8407155030382939 | `datasets` dataset `glue`, subset `stsb`, split `validation`|  

Footnote: For dataset STS-b, Successes = Pearson correlation, Accuracy = Spearman correlation

### `distilbert-base-uncased`

|           Model         |           Dataset   		   |        Dataset Descriptor          |      Successes      |       Accuracy      |  Technical Details                                                   |
| :---------------------: | :----------------------------: | :--------------------------------: | :-----------------: | :-----------------: | :------------------------------------------------------------------: |
|`distilbert-base-uncased`| AG News                        |  `distilbert-base-uncased-ag-news` |  944/1000           |  94.40%             | `datasets` dataset `ag_news`, split `test`                           |
|                         | CoLA                           |  `distilbert-base-uncased-cola`    |  786/1000           |  78.60%             | `datasets` dataset `glue`, subset `cola`, split `validation`         |
|                         | IMDB                           |  `distilbert-base-uncased-imdb`    |  903/1000           |  90.30%             | `datasets` dataset `imdb`, split `test`                              |
|                         | MNLI matched                   |  `distilbert-base-uncased-mnli`    |  817/1000           |  81.70%             | `datasets` dataset `glue`, subset `mnli`, split `validation_matched` |
|                         | MRPC                           |  `distilbert-base-uncased-mrpc`    |  350/408            |  85.78%             | `datasets` dataset `glue`, subset `mrpc`, split `validation`         |
|                         | QNLI                           |  `distilbert-base-uncased-qnli`    |  860/1000           |  86.00%             | `datasets` dataset `glue`, subset `qnli`, split `validation`         |
|                         | Recognizing Textual Entailment |  `distilbert-base-uncased-rte`     |  180/277            |  64.98%             | `datasets` dataset `glue`, subset `rte`, split `validation`          |
|                         | STS-b                          |  `distilbert-base-uncased-stsb`    |  0.8421540899520146 |  0.8407155030382939 | `datasets` dataset `glue`, subset `stsb`, split `validation`         |
|                         | WNLI                           |  `distilbert-base-uncased-wnli`    |  40/71              |  56.34%             | `datasets` dataset `glue`, subset `wnli`, split `validation`         |

Footnote: For dataset STS-b, Successes = Pearson correlation, Accuracy = Spearman correlation

### `roberta-base`

|          Model        |           Dataset   		     |   Dataset Descriptor   |      Successes     |       Accuracy      |  Technical Details                                           |
| :-------------------: | :----------------------------: | :--------------------: | :----------------: | :-----------------: | :----------------------------------------------------------: |
| `roberta-base`        | AG News                        | `roberta-base-ag-news` |  947/1000          |  94.70%             | `datasets` dataset `ag_news`, split `test`                   |
|                       | CoLA                           | `roberta-base-cola`    |  857/1000          |  85.70%             | `datasets` dataset `glue`, subset `cola`, split `validation` |
|                       | IMDB                           | `roberta-base-imdb`    |  941/1000          |  94.10%             | `datasets` dataset `imdb`, split `test`                      |
|                       | Movie Reviews [Rotten Tomatoes]| `roberta-base-mr`      |  899/1000          |  89.90%             | `datasets` dataset `rotten_tomatoes`, split `validation`     |
|                       | Movie Reviews [Rotten Tomatoes]| `roberta-base-mr`      |  883/1000          |  88.30%             | `datasets` dataset `rotten_tomatoes`, split `test`           |
|                       | MRPC                           | `roberta-base-mrpc`    |  371/408           |  91.18%             | `datasets` dataset `glue`, subset `mrpc`, split `validation` |
|                       | QNLI                           | `roberta-base-qnli`    |  917/1000          |  91.70%             | `datasets` dataset `glue`, subset `qnli`, split `validation` |
|                       | Recognizing Textual Entailment | `roberta-base-rte`     |  217/277           |  78.34%             | `datasets` dataset `glue`, subset `rte`, split `validation`  |
|                       | SST-2                          | `roberta-base-sst2`    |  820/872           |  94.04%             | `datasets` dataset `glue`, subset `sst2`, split `validation` |
|                       | STS-b                          | `roberta-base-stsb`    |  0.906067852162708 |  0.9025045272903051 | `datasets` dataset `glue`, subset `stsb`, split `validation` |
|                       | WNLI                           | `roberta-base-wnli`    |  40/71             |  56.34%             | `datasets` dataset `glue`, subset `wnli`, split `validation` |

Footnote: For dataset STS-b, Successes = Pearson correlation, Accuracy = Spearman correlation

### `xlnet-base-cased`

|          Model        |           Dataset   		     |    Dataset Descriptor    |      Successes     |       Accuracy      |  Technical Details                                           |
| :-------------------: | :----------------------------: | :----------------------: | :----------------: | :-----------------: | :----------------------------------------------------------: |
| `xlnet-base-cased`    | CoLA                           | `xlnet-base-cased-cola`  |  800/1000          |  80.00%             | `datasets` dataset `glue`, subset `cola`, split `validation` |
|                       | IMDB                           | `xlnet-base-cased-imdb`  |  957/1000          |  95.70%             | `datasets` dataset `imdb`, split `test`                      |
|                       | Movie Reviews [Rotten Tomatoes]| `xlnet-base-cased-mr`    |  908/1000          |  90.80%             | `datasets` dataset `rotten_tomatoes`, split `validation`     |
|                       | Movie Reviews [Rotten Tomatoes]| `xlnet-base-cased-mr`    |  876/1000          |  87.60%             | `datasets` dataset `rotten_tomatoes`, split `test`           |
|                       | MRPC                           | `xlnet-base-cased-mrpc`  |  363/408           |  88.97%             | `datasets` dataset `glue`, subset `mrpc`, split `validation` |
|                       | Recognizing Textual Entailment | `xlnet-base-cased-rte`    |  196/277           |  70.76%             | `datasets` dataset `glue`, subset `rte`, split `validation`  |
|                       | STS-b                          | `xlnet-base-cased-stsb`   |  0.883111673280641 |  0.8773439961182335 | `datasets` dataset `glue`, subset `stsb`, split `validation` |
|                       | WNLI                           | `xlnet-base-cased-wnli`   |  41/71             |  57.75%             |  `datasets` dataset `glue`, subset `wnli`, split `validation`|

Footnote: For dataset STS-b, Successes = Pearson correlation, Accuracy = Spearman correlation
