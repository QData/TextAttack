#!/bin/bash
# Trains `bert-base-cased` on the STS-B task for 3 epochs. This is a basic
# demonstration of our training script and `datasets` integration.
textattack train --model-name-or-path lstm --dataset imdb  --epochs 50 --learning-rate 1e-5