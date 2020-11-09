#!/bin/bash
# Trains `bert-base-cased` on the STS-B task for 3 epochs. This is a basic
# demonstration of our training script and `datasets` integration.
textattack train --model lstm --dataset rotten_romatoes --batch-size 64 --epochs 50 --learning-rate 1e-5