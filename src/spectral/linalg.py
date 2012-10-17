"""
Linear algebra methods for spectral learning
"""

import scipy as sc 
from scipy import array
from scipy.linalg import svd, svdvals

def apply_permutation( perm, lst ):
    """Apply a permutation to a list"""
    return [ lst[ i ] for i in perm ]

def invert_permutation( perm ):
    """Invert a permutation @perm"""
    perm_ = [ i for i in xrange( len( perm ) ) ]
    for (i, j) in zip( xrange( len( perm ) ), perm ):
        perm_[j] = i
    return perm_

def apply_matrix_permutation( perm, x ):
    """Apply a permutation to a matrix"""
    rowp, colp = perm
    # Permute the columns in a row
    if colp is not None:
        x = array( map( lambda row: apply_permutation( colp, row ),
            x.tolist() ) )
    # Permute the rows in a column
    if rowp is not None:
        x = array( map( lambda col: apply_permutation( rowp, col ),
            x.T.tolist() ) ).T
    return x

def invert_matrix_permutation( perm ):
    """Invert a matrix permutation @perm"""
    rowp, colp = perm
    return invert_permutation( rowp ), invert_permutation( colp )

def canonical_permutation( x ):
    """Rearrange the rows and columns of a matrix so that the maximum
    element is at (0,0), the elements of the first row are in sorted
    order and the first columns is in sorted order of the rows.
    Returns the permutation required to make it"""

    # Find the smallest element.
    idx = x.argmin()
    r, c = sc.unravel_index( idx, x.shape )
    # Apply the sorts that bring x to the top
    colp = x[r, :].argsort()
    rowp = x[:, c].argsort()
    perm = (rowp, colp)

    return perm

def canonicalise( x ):
    """Return the version permutation of x"""
    perm = canonical_permutation( x )
    return apply_matrix_permutation( perm, x )

def svdk( x, k ):
    """Top-k SVD decomposition"""
    U, D, Vt = svd( x, full_matrices=False )
    return U[:, :k], D[:k], Vt[:k, :]

def mrank( x, eps=1e-12 ):
    """Matrix rank"""
    d = svdvals( x )
    return len( [v for v in d if abs(v) > eps ] ) 

def condition_number( x, k = None ):
    """Condition number for the k-rank approximation of x"""
    # Get the eigenvalues
    s = svdvals( x )

    if k is not None:
        return s[0]/s[k]
    else:
        return s[0]/s[-1]
    
def eigengap( x, k = None ):
    """Minimum difference in eigenvalues"""
    # Get the eigenvalues
    s = svdvals( x )

    return sc.diff( s ).min() / s[0]

