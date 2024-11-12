import random

import pandas as pd
import numpy as np
from tqdm import trange


def sample_feature_subset(N, n, count):
    print(f"Generating {count} samples of {n} features from a total of {N} features ...")
    ls_seq = []
    for _ in trange(count):
        seq = np.zeros((N, ))

        indices = random.sample(range(N), n)
        seq[indices] = 1
        ls_seq.append(list(seq))
    return ls_seq


def hamming_distance(seq1, seq2):
    dis = np.sum(np.abs(seq1 - seq2))
    return dis
























