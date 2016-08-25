#!/bin/bash
echo 'Determine coverage of all unittests...'

mkdir -p docs/coverage

nosetests-2.7 --with-coverage \
    --cover-erase --cover-inclusive \
    --cover-package=utils \
    --cover-html --cover-html-dir="docs/coverage"
    