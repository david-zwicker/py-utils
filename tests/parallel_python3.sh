#!/bin/bash

CORES=`python -c 'import multiprocessing as mp; print(mp.cpu_count())'`

echo 'Run unittests on '$CORES' cores...'
cd ..
nosetests-3.4 --processes=$CORES --process-timeout=60 --stop
