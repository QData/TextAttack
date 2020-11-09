#!/bin/bash
# Shows how to attack a DistilBERT model fine-tuned on SST2 dataset *from the
# huggingface model repository& using the DeepWordBug recipe and 10 examples.
textattack attack --model-from-huggingface distilbert-base-uncased-finetuned-sst-2-english --dataset-from-huggingface glue^sst2 --recipe deepwordbug --num-examples 10
