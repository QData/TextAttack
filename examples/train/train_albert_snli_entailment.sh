#!/bin/bash
# Trains `bert-base-cased` on the STS-B task for 3 epochs. This is a 
# demonstration of how our training script can handle different `transformers`
# models and customize for different datasets.
textattack train --model albert-base-v2 --dataset snli --batch-size 128 --epochs 5 --max-length 128 --learning-rate 1e-5 --allowed-labels 0 1 2