#!/bin/bash
echo 'Determine coverage of all unittests...'

cd ..
mkdir -p tests/coverage

nosetests --with-coverage \
    --cover-erase --cover-inclusive \
    --cover-package=utils \
    --cover-html --cover-html-dir="tests/coverage"
