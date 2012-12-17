"""
Phase recovery using proximal subgradient descent
"""

import ipdb
import scipy as sc
from scipy import diag, array, ndim, outer, eye, ones, log, sqrt, zeros, floor, exp
from scipy.linalg import norm, svd, svdvals, eig, eigvals, inv, det, cholesky
rand, randn = sc.rand, sc.randn

import matplotlib.pyplot as plt

def gradient_step( y, X, B ):
    """
    Take a gradient step along each B to optimise 
    $\diff{L}{B_{ij}} = \sum_i (x^i' B x^i - y^i^2) x^i_i x^i_j

    Including the constraint that $B$ is symmetric, this becomes,
    $\diff{L}{B_{ij}} = 2 \sum_i (x^i' B x^i - y^i^2) x^i_i x^i_j
    """ 

    d, _ = B.shape
    N = len(y)

    # Compute x^T B x - y
    dB = zeros( (d, d) )
    for (y_i, x_i) in zip( y, X ):
        dB += (x_i.T.dot( B ).dot( x_i ) - y_i**2) * outer( x_i, x_i )

    return 2 * dB/N

def nuclear_subgradient( X ):
    """Make X low rank"""
    U, _, Vt = svd( X, full_matrices = False )

    return U.dot( Vt )

def low_rankify( X, thresh ):
    """Make X low rank"""
    U, S, Vt = svd( X )
    S -= thresh
    S[ S < 0.0 ] = 0.0
    return U.dot( diag( S ) ).dot( Vt )

def finite_norm( X, bound = 100.0 ):
    """Make X low rank"""
    scale = norm( X )/bound
    if scale > 1.0:
        X = X / scale
    return X

def residual( y, X, B ):
    """Residual in error"""
    N = len( y )
    tot = 0
    for (x_i, y_i) in zip( X, y ):
        tot += (x_i.T.dot( B ).dot( x_i ) - y_i**2)**2
    return tot/N

def solve( y, X, B0 = None, reg = 1e-2, iters = 100, alpha = "1/T" ):
    """
    Solve for B that optimises the following objective:
    $\\argmin_{B} = 1/2 * \sum_i (x_i^T B x_i - y_i^2)^2/N + \lambda Tr(B)$
    Every row of X corresponds to a y
    """

    N, d = X.shape

    # Precompute X X^T

    # Initialise the B0
    if B0 is None:
        B0 = zeros( (d, d) ) 

    # Gradient norm
    gradient_norms = []
    # Residual norm
    residuals = []

    B_ = B0
    for i in xrange( iters ):
        # Run a gradient descent step
        residuals.append( residual( y, X, B_ ) )
        dB = gradient_step( y, X, B_ )
        # Add the subgradient
        dB += reg * nuclear_subgradient( B_ )

        gradient_norms.append( norm( dB ) )
        print i, residuals[-1], gradient_norms[-1]

        if alpha == "1/T" :
            alpha = 0.1/sqrt(i+1)
        B = B_ - alpha * dB

        # Do the proximal step of making B low rank by soft thresholding k.
        B_ = low_rankify( B_, reg )

        # Proximal step of constraining to bounded norm
        B = finite_norm( B )

        # Check convergence
        if norm( B - B_ ) / norm( B ) < 1e-7:
            break
        B_ = B

    # Plot graphs
    plt.plot( residuals )
    plt.plot( gradient_norms )
    plt.show()

    return B


def test_solve():
    N = 10000
    d = 3 
    k = 2

    pi = array( [0.5, 0.5] ) 
    B = eye( d, k )
    B2 = B.dot( diag( pi ) ).dot( B.T )
    X = randn( N, d )
    y = array( [ x.T.dot( B2 ).dot( x ) for x in X ] )

    print residual( y, X, B2 )

    #B20 = B2 + 0.1 * randn( d, d )
    #B2_ = solve( y, X, B20 )
    B2_ = solve( y, X, B2, alpha = 0.1 )

    print B2_, B2
    print svdvals(B2_), svdvals(B2)
    print norm(B2 - B2_)/norm(B2)

if __name__ == "__main__":
    test_solve()


