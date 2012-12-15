"""
Phase recovery using proximal subgradient descent
"""

import scipy as sc
from scipy import diag, array, ndim, outer, eye, ones, log, sqrt, zeros, floor, exp
from scipy.linalg import norm, svd, svdvals, eig, eigvals, inv, det, cholesky
rand, randn = sc.rand, sc.randn

def gradient_step( y, X, B, iter, alpha = "1/T" ):
    """
    Take a gradient step along each B to optimise 
    $\diff{L}{B_{ij}} = \sum_i (x^i' B x^i - y^i^2) x^i_i x^i_j

    Including the constraint that $B$ is symmetric, this becomes,
    $\diff{L}{B_{ij}} = 2 \sum_i (x^i' B x^i - y^i^2) x^i_i x^i_j
    """ 

    d, _ = B.shape

    # Compute x^T B x - y
    dB = zeros( (d, d) )
    for (y_i, x_i) in zip( y, X ):
        dB += (x_i.T.dot( B ).dot( x_i ) - y_i**2) * outer( x_i, x_i )
    if alpha == "1/T" :
       alpha = 1.0/(iter+1)
    B -= alpha * dB/len(y)

    return B

def low_rankify( X, thresh = 1.0 ):
    """Make X low rank"""
    U, S, Vt = svd( X )
    S[ S < thresh ] = 0.0
    return U.dot( diag( S ) ).dot( Vt )

def solve( y, X, B0 = None, coeff = 1e-2, iters = 100 ):
    """
    Solve for B that optimises the following objective:
    $\\argmin_{B} = 1/2 * \sum_i (x_i^T B x_i - y_i^2)^2 + \lambda Tr(B)$
    Every row of X corresponds to a y
    """

    d, _ = X.shape

    # Precompute X X^T

    # Initialise the B0
    if B0 is None:
        B0 = eye( d )

    B = B0
    for i in xrange( iters ):
        # Run a gradient descent step
        print B
        B_ = gradient_step( y, X, array(B), i )

        # Do the proximal step of making B low rank by soft thresholding k.
        B_ = low_rankify( B_ )

        # Check convergence
        print norm( B - B_ )
        B = B_
    return B


def test_solve():
    N = 1000
    d = 3 
    k = 2

    pi = array( [0.5, 0.5] ) 
    B = eye( d, k )
    B2 = B.dot( diag( pi ) ).dot( B.T )
    X = randn( N, d )
    y = array( [ x.T.dot( B2 ).dot( x ) for x in X ] )

    B20 = B2 + 0.1 * randn( d, d )

    B2_ = solve( y, X, B20 )

    print svdvals(B2_)
    print svdvals(B2)
    print norm(B2 - B2_)

if __name__ == "__main__":
    test_solve()


