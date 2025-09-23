import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

def get_mlp(output_size):

    model = nn.Sequential(
        nn.Linear(768, 512),
        # nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Linear(512, 512),
        # nn.BatchNorm1d(1024),
        nn.ReLU(),
        # nn.Dropout(p=0.3),
        nn.Linear(512, 256),
        # nn.BatchNorm1d(512),
        nn.ReLU(),
        # nn.Dropout(p=0.3),
        nn.Linear(256, 128),
        # nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(128, output_size),
    )
    
    return model

class Embeddings_transcripts_dataset(Dataset):
    def __init__(self, embeddings, gene_expressions):
        """
        embeddings: (num_samples, embedding_dim)
        gene_expressions: (num_samples, num_genes)
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.gene_expressions = torch.tensor(gene_expressions, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.gene_expressions[idx]

class Embeddings_transcripts_dataset_ids(Dataset):
    def __init__(self, embeddings, gene_expressions, ids):
        """
        embeddings: (num_samples, embedding_dim)
        gene_expressions: (num_samples, num_genes)
        ids: (num_samples)
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.gene_expressions = torch.tensor(gene_expressions, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.gene_expressions[idx], self.ids[idx]

import numpy as np

def scale(arr, mean=None, std=None):
    """
    Perform Z-scoring (standardization) on a 2D NumPy array.
    Centers each column to zero mean and scales to unit variance.
    
    Parameters:
        arr (numpy.ndarray): Input 2D array (rows are samples, columns are features).
        mean (numpy.ndarray, optional): Precomputed mean values for each column.
        std (numpy.ndarray, optional): Precomputed standard deviation values for each column.
    
    Returns:
        numpy.ndarray: Z-scored array.
        numpy.ndarray: Mean values used for scaling.
        numpy.ndarray: Standard deviation values used for scaling.
    """
    if mean is None:
        mean = np.mean(arr, axis=0)
    if std is None:
        std = np.std(arr, axis=0, ddof=0)
    
    std[std == 0] = 1  # Avoid division by zero
    
    return (arr - mean) / std, mean, std

def undo_standard_scaling(scaled_array, means, stds):
    """
    Reverse standard scaling transformation for a NumPy array.

    Parameters:
    - scaled_array: NumPy array of scaled values (shape: [n_samples, n_features])
    - means: NumPy array of original means (shape: [n_features])
    - stds: NumPy array of original standard deviations (shape: [n_features])

    Returns:
    - NumPy array of unscaled values
    """
    return (scaled_array * stds) + means

def row_norm_min_max(array):
    """
    Row-wise min-max scaling transformation for a NumPy array.

    Parameters:
    - array: 2D NumPy array (shape: [n_samples, n_features])

    Returns:
    - NumPy array of row-wise min-max values
    """
    row_min = np.min(array, axis=1, keepdims=True)
    row_max = np.max(array, axis=1, keepdims=True)
    return (array - row_min) / (row_max - row_min)

def undo_row_norm(scaled_array, mins, maxs):
    """
    Reverse a row-wise min-max scaling transformation for a NumPy array.

    Parameters:
    - scaled_array: NumPy array of scaled values (shape: [n_samples, n_features])
    - mins: NumPy array of row-wise minima (shape: [n_features])
    - maxs: NumPy array of row-wise maxima (shape: [n_features])

    Returns:
    - NumPy array of unscaled values
    """
    return (scaled_array * (maxs-mins)) + mins
    