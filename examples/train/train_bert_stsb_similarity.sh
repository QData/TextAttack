#!/bin/bash
# Trains `bert-base-cased` on the STS-B task for 3 epochs. This is a demonstration
# of how our training script handles regression.
textattack train --model bert-base-cased --dataset glue^stsb --batch-size 128 --epochs 3 --max-length 128 --learning-rate 1e-5