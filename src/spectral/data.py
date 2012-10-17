"""
Data manipulation functions
"""

import scipy as sc
from scipy import matrix, array
from scipy.linalg import norm 

from . import _data

Pairs = _data.Pairs

def Triples( x1, x2, x3, eta = None):
    """Compute E[x1 \ctimes x2 <eta, x3]
       If eta is None, return a lambda function that takes an eta and produces a result.
    """
    if eta is not None:
        return _data.Triples( x1, x2, x3, eta )
    else:
        return lambda eta: _data.Triples( x1, x2, x3, eta )

