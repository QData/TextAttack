#!/bin/bash
# Shows how to attack our RoBERTA model fine-tuned on SST2 using the TextFooler
# recipe and 10 examples.
textattack attack --model roberta-base-sst2 --recipe textfooler --num-examples 10