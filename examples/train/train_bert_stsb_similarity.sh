#!/bin/bash
# Trains `bert-base-cased` on the STS-B task for 3 epochs. This is a demonstration
# of how our training script handles regression.
textattack train --model-name-or-path bert-base-cased --dataset glue^stsb  --epochs 3 --learning-rate 1e-5