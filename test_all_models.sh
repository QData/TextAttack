#!/bin/bash

python textattack/run_attack.py --model lstm-yelp-sentiment --data yelp --num_example=2
python textattack/run_attack.py --model bert-yelp-sentiment --data yelp --num_examples=2
python textattack/run_attack.py --model cnn-yelp-sentiment --data yelp --num_examples=2

