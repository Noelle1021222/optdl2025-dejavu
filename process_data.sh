#!/bin/bash

SOURCE_DIR="raw_data"
# folder to save svm data format
SAVE_SVM_DATA_DIR="svm_data"
DATASETS=("aloi" "mnist")

# process libsvm
python3 process_libsvm.py --datasets "${DATASETS[@]}" --source_folder "$SOURCE_DIR" --save_folder "$SAVE_SVM_DATA_DIR"

