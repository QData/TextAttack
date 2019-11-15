#!/bin/bash
#
# Tests our models on a variety of datasets.
#
# Usage: ./test_all_models.sh <dataset>
# datasets: imdb, yelp, mr
#
if [ $# -eq 0 ]; then
    echo "No arguments provided. Run script like  './test_all_models.sh <dataset>'."
    exit 1
fi

if [ $1 = "mr" ] || [ $1 = "all" ]; then
  echo "";
  echo "Testing models on dataset mr.";
  python textattack/run_attack.py --model bert-mr --data mr --num_examples=2;
  python textattack/run_attack.py --model lstm-mr --data mr --num_example=2;
  python textattack/run_attack.py --model cnn-mr --data mr  --num_examples=2;
fi

if [ $1 = "imdb" ] || [ $1 = "all" ]; then
  echo "";
  echo "Testing models on dataset imdb."
  python textattack/run_attack.py --model bert-imdb --data imdb --num_examples=2;
  python textattack/run_attack.py --model lstm-imdb --data imdb --num_example=2;
  python textattack/run_attack.py --model cnn-imdb --data imdb --num_examples=2;
fi

if [ $1 = "yelp" ] || [ $1 = "all" ]; then
  echo "";
  echo "Testing models on dataset yelp."
  python textattack/run_attack.py --model bert-yelp-sentiment --data yelp-sentiment --num_examples=2;
  python textattack/run_attack.py --model lstm-yelp-sentiment --data yelp-sentiment --num_example=2;
  python textattack/run_attack.py --model cnn-yelp-sentiment --data yelp-sentiment --num_examples=2;
fi
