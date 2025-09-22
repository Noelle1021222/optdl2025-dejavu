#!/bin/bash

# download directory
DOWNLOAD_DIR="raw_data"
mkdir -p "$DOWNLOAD_DIR"

echo "Downloading..."
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/aloi.scale.bz2 -O  "$DOWNLOAD_DIR/aloi.scale.bz2"
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2 -O "$DOWNLOAD_DIR/mnist.scale.bz2"

echo "Decompressing..."
bzip2 -d  "$DOWNLOAD_DIR/aloi.scale.bz2"
bzip2 -d  "$DOWNLOAD_DIR/mnist.scale.bz2"