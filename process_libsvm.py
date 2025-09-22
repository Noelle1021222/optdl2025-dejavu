import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import StratifiedKFold

import warnings
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_libsvm_data(file_path):
    labels = []
    features = []
    with open(file_path, 'r') as f:
        for line in f:
            elements = line.strip().split()
            labels.append(int(elements[0]))  
            feature_dict = {int(k.split(':')[0]): float(k.split(':')[1]) for k in elements[1:]}
            features.append(feature_dict)  
    return np.array(labels), features


def process_libsvm(dataset_name, source_folder='raw_data', save_folder='svm_data'):

    save_dir = os.path.join(save_folder, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    labels, features = parse_libsvm_data(f'{source_folder}/{dataset_name}.scale')

    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(features)

    scv5 = StratifiedKFold(n_splits=5)

    for fold_idx, (train_index, test_index) in enumerate(scv5.split(X, labels),1):
        trainX, testX = X[train_index], X[test_index]
        trainY, testY = labels[train_index], labels[test_index]
        
        dump_svmlight_file(trainX, trainY, os.path.join(save_dir, f'train_{dataset_name}_{fold_idx}.svm'), zero_based=False)
        dump_svmlight_file(testX, testY , os.path.join(save_dir, f'test_{dataset_name}_{fold_idx}.svm'), zero_based=False)
        print(f"Train fold size: {len(train_index)}, Test fold size: {len(test_index)}")
    print('Preprocessing finished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets and save them in SVM format.")
    parser.add_argument("--datasets", nargs='+', required=True, help="List of dataset names.")
    parser.add_argument("--source_folder", type=str, required=True, help="Directory of raw data.")
    parser.add_argument("--save_folder", type=str, required=True, help="Directory to save processed SVM files.")
    args = parser.parse_args()

    for dataset_name in args.datasets:
        print('Dataset name:', dataset_name)
        process_libsvm(dataset_name, source_folder=args.source_folder, save_folder=args.save_folder)