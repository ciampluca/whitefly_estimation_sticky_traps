#!/bin/bash

DATASET_ROOT="./"

if [[ -z "$DATASET_ROOT" ]]; then
    DATASET_ROOT="./"
fi

DATA_DIR="${DATASET_ROOT}/sticky-trap-insects"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Downloading and extracting: Pest Sticky Traps dataset"
    zenodo_get 10.5281/zenodo.7801239       # zenodo DOI
    unzip pest-sticky-traps.zip -d "${DATA_DIR}"  && rm pest-sticky-traps.zip && rm md5sums.txt
fi

	