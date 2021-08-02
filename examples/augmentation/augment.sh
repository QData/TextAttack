#!/bin/bash
textattack augment --input-csv examples.csv --output-csv output.csv --input-column text --recipe eda --pct-words-to-swap .1 --transformations-per-example 2 --exclude-original --overwrite
