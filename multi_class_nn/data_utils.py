import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, Subset
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import StratifiedShuffleSplit


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define a custom dataset using LIBSVM's svm_read_problem
class SVMFormatDataset(Dataset):
    def __init__(self, file_path, num_features=None):
        data, labels = load_svmlight_file(file_path)
        data = torch.tensor(data.toarray(), dtype=torch.float32)
        if num_features is not None:
            resize_data = torch.zeros((len(labels), num_features))
            available_dim = min(num_features, data.shape[1])
            resize_data[:,:available_dim] = data[:,:num_features]
            self.data = resize_data
        else:
            self.data = data
        if min(labels) == 1:
            self.labels = torch.tensor(labels - 1, dtype=torch.long)  # Convert labels to 0-based indexing
        else :
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_num_features(self):
        return self.data.shape[1]

# Dataset preprocessing for OVR
def prepare_ovr_data(dataset, target_label):
    data, labels = [], []
    for features, label in dataset:
        data.append(features)
        labels.append(1 if label == target_label else 0)
    return torch.stack(data), torch.tensor(labels)

# Dataset preprocessing for OVO
def prepare_ovo_data(dataset, label1, label2):
    data, labels = [], []
    for features, label in dataset:
        if label == label1 or label == label2:
            data.append(features)
            labels.append(1 if label == label1 else 0)
    return torch.stack(data), torch.tensor(labels)

def stratified_split(dataset, test_size=0.2, seed=42):

    labels = dataset.labels.numpy()
    label_counts = Counter(labels)
    rare_classes = {label for label, count in label_counts.items() if count < 3}
    rare_indices = [i for i, label in enumerate(labels) if label in rare_classes]
    non_rare_indices = [i for i, label in enumerate(labels) if label not in rare_classes]
    non_rare_labels = [labels[i] for i in non_rare_indices]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(non_rare_indices, non_rare_labels))

    rare_train_indices = rare_indices[:len(rare_indices)//2]
    rare_val_indices = rare_indices[len(rare_indices)//2:]

    train_indices = [non_rare_indices[i] for i in train_idx] + rare_train_indices
    val_indices = [non_rare_indices[i] for i in val_idx] + rare_val_indices

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset  