#!/bin/bash
echo 'Determine coverage of all unittests...'

cd ..
mkdir -p tests/coverage_python2

nosetests-2.7 --with-coverage \
    --cover-erase --cover-inclusive \
    --cover-package=utils \
    --cover-html --cover-html-dir="tests/coverage_python2"
