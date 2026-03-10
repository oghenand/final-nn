# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
# MY IMPORTS
import random

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    data_len = len(labels)
    # first find minority class
    minority_class = 1 if sum(labels) <=data_len/2 else 0
    # find number to sample to balance
    n_sample = len([i for i in labels if i != minority_class]) - len([i for i in labels if i == minority_class])
    # get idxs to sample from - create set for faster (O(1)) look up
    idx_sample = random.choices(
        [i for i, j in enumerate(labels) if j == minority_class], 
        k=n_sample
    )
    sampled_labels = [labels[i] for i in idx_sample]
    sampled_seqs = [seqs[i] for i in idx_sample]
    # attach to original lists
    sampled_seqs = seqs + sampled_seqs
    sampled_labels = labels + sampled_labels
    # shuffle since end of seqs/labels are all minority class
    combined = list(zip(sampled_seqs, sampled_labels))
    random.shuffle(combined)
    sampled_seqs, sampled_labels = zip(*combined)
    sampled_seqs, sampled_labels = list(sampled_seqs), list(sampled_labels)
    # return
    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # initialize encoding based off what each base pair should match to
    base_encode = {
        'A': 0,
        'T': 1,
        'C': 2,
        'G': 3,
    }
    encode_mapping = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
    # iterate w/ numpy
    # first create list of encoding indices to extract (ie which element of encode_mapping to use)
    # then use numpy to grab all the sequences in one shot
    encodings = [
        encode_mapping[[base_encode[base] for base in seq]].flatten() for seq in seq_arr
    ]
    return encodings