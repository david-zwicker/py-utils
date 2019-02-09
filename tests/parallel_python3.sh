#!/bin/bash

CORES=`python -c 'from multiprocessing import cpu_count; print(cpu_count() // 2)'`

echo 'Run unittests on '$CORES' cores...'
python3 -m pytest -n $CORES ..
