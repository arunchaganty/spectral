"""
Data manipulation functions
"""

import scipy as sc
from scipy import matrix, array
from scipy.linalg import norm 

from . import _data

def count_frequency( X, d ):
    N, W = X.shape
    Y = _data.count_frequency( X, d )
    return Y/float(W)

Pairs = _data.Pairs
Triples = _data.Triples

