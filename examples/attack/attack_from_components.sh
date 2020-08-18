#!/bin/bash
# Shows how to build an attack from components and use it on a pre-trained
# model on the Yelp dataset.
textattack attack --attack-n --goal-function untargeted-classification \
    --model bert-base-uncased-yelp --num-examples 8 --transformation word-swap-wordnet \
    --constraints edit-distance^12 max-words-perturbed:max_percent=0.75 repeat stopword \
    --search greedy