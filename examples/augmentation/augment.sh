#!/bin/bash
textattack augment --csv examples.csv --input-column text --recipe embedding --num-words-to-swap 4 --transformations-per-example 2 --exclude-original
