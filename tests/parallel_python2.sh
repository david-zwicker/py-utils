#!/bin/bash

CORES=`python -c 'from multiprocessing import cpu_count; print(cpu_count())'`

echo 'Run unittests on '$CORES' cores...'
cd ..
nosetests-2.7 --processes=$CORES --process-timeout=60 --stop
