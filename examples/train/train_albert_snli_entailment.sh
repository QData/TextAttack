#!/bin/bash
# Trains `bert-base-cased` on the STS-B task for 3 epochs. This is a 
# demonstration of how our training script can handle different `transformers`
# models and customize for different datasets.
textattack train --model-name-or-path albert-base-v2 --dataset snli --per-device-train-batch-size 8 --epochs 5 --learning-rate 1e-5 