#!/bin/bash

echo 'cool stuff happening'

shopt -s globstar

for f in ./**/*.py; do

    docformatter --in-place "$f"
    autopep8 -i -a -a --select=E22,E266 "$f"

done

echo 'no more cool stuff'
