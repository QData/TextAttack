#!/bin/bash
textattack augment --csv examples.csv --input-column text --recipe eda --pct-words-to-swap .1 --transformations-per-example 2 --exclude-original --overwrite
