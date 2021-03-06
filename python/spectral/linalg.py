"""
Linear algebra methods for spectral learning
"""

import scipy as sc 
from scipy import array, diag
from scipy.linalg import svd, svdvals, norm, eigvals, eig
from spectral.rand import orthogonal

from spectral.munkres import Munkres
from spectral.hosvd import unfold, fold, HOSVD

#from spectral import _data
#apply_shuffle = _data.apply_shuffle

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

def svdk( X, k ):
    """Top-k SVD decomposition"""
    U, D, Vt = svd( X, full_matrices=False )
    return U[:, :k], D[:k], Vt[:k, :]

def approxk( X, k ):
    """Best k rank approximation of X"""
    U, D, Vt = svdk( X, k )
    return U.dot( diag( D ) ).dot( Vt )

def mrank( x, eps=1e-12 ):
    """Matrix rank"""
    d = svdvals( x )
    return len( [v for v in d if abs(v) > eps ] ) 

def condition_number( x, k = None ):
    """Condition number for the k-rank approximation of x"""
    # Get the eigenvalues
    s = svdvals( x )

    if k is not None:
        return s[0]/s[k-1]
    else:
        return s[0]/s[-1]
    
def column_gap( X, k ):
    """Minimum difference between column values"""
    mag = 0
    diff = sc.inf
    for i in xrange(k):
        mag_ = norm( X.T[i] )
        if mag_ > mag:
            mag = mag_
        for j in xrange(i+1, k):
            diff_ = norm( X.T[i] - X.T[j] )
            if diff_ < diff:
                diff = diff_
    return diff/mag
    
def column_sep( X ):
    """Minimum difference in and between column values"""
    d,k = X.shape
    mag = sc.inf
    diff = sc.inf

    for i in xrange(k):
        mag_ = norm( X.T[i] )
        if mag_ < mag:
            mag = mag_
        for j in xrange(i+1, k):
            diff_ = norm( X.T[i] - X.T[j] )
            if diff_ < diff:
                diff = diff_
    return min( diff, mag )
    
def spectral_gap( x, k = None ):
    """Minimum difference in eigenvalues"""
    # Get the singular values
    s = svdvals( x )
    if k is not None:
        s = s[:k]

    return (sc.diff( s )).min() / s[0]

def eigen_sep( X, k = None ):
    """Minimum difference in eigenvalues"""
    # Get the eigenvalues
    s = (eigvals( X )).real
    if k is not None:
        s = s[:k]

    return min(abs(s).min(), abs(sc.diff( s )).min())

def column_aerr( M, M_ ):
    return max( map( lambda (mu, mu_): norm( mu - mu_ ),  zip( M.T, M_.T
        ) ) )

def column_rerr( M, M_ ):
    return max( map( lambda (mu, mu_): norm( mu - mu_ )/norm( mu ),  zip(
        M.T, M_.T ) ) )

def tensor_norm( T, d, ntype=2 ):
    """Approximately calculate a tensor norm by projecting it onto a number of vectors"""
    theta = orthogonal( d )

    return max( [ norm( T(t), ntype ) for t in theta ] )

def tensor_aerr( T, T_, d, ntype=2 ):
    """Approximately calculate aerr of a tensor by projecting it onto a number of vectors"""
    theta = orthogonal( d )
    return max( [norm( T(t) - T_(t), ntype ) for t in theta ] )

def tensor_rerr( T, T_, d, ntype=2 ):
    """Approximately calculate rerr of a tensor by projecting it onto a number of vectors"""
    theta = orthogonal( d )
    return max( [norm( T(t) - T_(t), ntype )/norm( T(t), ntype ) for t in theta ] )

def closest_permuted_vector( a, b ):
    """Find a permutation of b that matches a most closely (i.e. min |A
    - B|_2)"""

    # The elements of a and b form a weighted bipartite graph. We need
    # to find their minimal matching.
    assert( a.shape == b.shape )
    n, = a.shape

    W = sc.zeros( (n, n) )
    for i in xrange( n ):
        for j in xrange( n ):
            W[i, j] = (a[i] - b[j])**2

    m = Munkres()
    matching = m.compute( W )
    matching.sort()
    _, bi = zip(*matching)

    return bi

def closest_permuted_matrix( A, B ):
    """Find a _row_ permutation of B that matches A most closely (i.e. min |A
    - B|_F)"""

    # The rows of A and B form a weighted bipartite graph. The weights
    # are computed using the vector_matching algorithm.
    # We need to find their minimal matching.
    assert( A.shape == B.shape )

    n, _ = A.shape
    m = Munkres()

    # Create the weight matrix
    W = sc.zeros( (n, n) )
    for i in xrange( n ):
        for j in xrange( n ):
            # Best matching between A and B
            W[i, j] = norm(A[i] - B[j])
        
    matching = m.compute( W )
    matching.sort()
    _, rowp = zip(*matching)
    rowp = array( rowp )
    # Permute the rows of B according to Bi
    B_ = B[ rowp ]

    return B_

def test_closest_permuted_matrix():
    """Test whether the closest_permuted_matrix fn works"""
    A = array([[0, 1, 2], [2, 3, 4], [4, 5, 6]])
    B = array([[2.3, 3.1, 4.1], [4.1, 5.2, 5.9], [0.1, 1.3, 1.9]])
    Bo = array([[0.1, 1.3, 1.9], [2.3, 3.1, 4.1], [4.1, 5.2, 5.9]])

    B_ = closest_permuted_matrix( A, B )
    assert( sc.allclose( B_, Bo ) )

def tensorify( x1, x2, x3 ):
    """Create a tensor from x1 cross x2 cross x3"""
    M, = x1.shape
    N, = x2.shape
    O, = x3.shape
    assert( N == M and M == O )

    X = sc.outer(x1,x2)
    B = sc.zeros( (N, N, N) )
    for i in xrange(N):
        B[:,:,i] = X * x3[i]
    return B

def matrix_tensorify( A, x ):
    """Create a tensor from the matrix A and vector x"""
    N, M = A.shape
    O, = x.shape
    assert( N == M and M == O )

    B = sc.zeros( (N, N, N) )
    for i in xrange(N):
        B[:,:,i] = A * x[i]
    return B

def test_matrix_tensorify():
    """Test whether this tensorification routine works"""

    A = sc.eye( 3 )
    x = sc.random.rand(3)
    y = sc.ones( 3 )

    B = matrix_tensorify( A, x )

    assert ( B.dot( y )  == A * x.dot(y) ).all()
    C = B.swapaxes( 2, 0 )
    assert ( C.dot( y )  == sc.outer(x, y) ).all()
    D = B.swapaxes( 2, 1 )
    assert ( D.dot( y )  == sc.outer(y, x) ).all()

