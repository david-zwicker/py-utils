#!/bin/bash
echo 'Run unittests using python3...'
cd ..

# ignore deprecation warning since we also support python 2
python3 -W ignore::DeprecationWarning -m unittest discover
