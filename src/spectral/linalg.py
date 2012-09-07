"""
Linear algebra methods for spectral learning
"""

import scipy as m 

def apply_permutation( perm, lst ):
    """Apply a permutation to a list"""
    return [ lst[ i ] for i in perm ]

def invert_permutation( perm ):
    """Invert a permutation @perm"""
    perm_ = [ i for i in xrange( len( perm ) ) ]
    for (i,j) in zip( xrange( len( perm ) ), perm ):
        perm_[j] = i
    return perm_


