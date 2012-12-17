"""
Phase recovery using proximal subgradient descent
"""

import ipdb
import scipy as sc
from scipy import diag, array, ndim, outer, eye, ones, log, sqrt, zeros, floor, exp
from scipy.linalg import norm, svd, svdvals, eig, eigvals, inv, det, cholesky
rand, randn = sc.rand, sc.randn

from models import LinearRegressionsMixture

#import matplotlib.pyplot as plt

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
        dB += (x_i.T.dot( B ).dot( x_i ) - y_i) * outer( x_i, x_i )

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
        tot += (x_i.T.dot( B ).dot( x_i ) - y_i)**2
    return tot/N

def solve( y, X, B0 = None, reg = 1e-3, iters = 500, alpha = "1/T" ):
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
        dB = gradient_step( y, X, B_ )
        # Add the subgradient
        #dB += reg * nuclear_subgradient( B_ )

        if alpha == "1/T" :
            alpha = 0.01/sqrt(i+1)
        B = B_ - alpha * dB

        # Do the proximal step of making B low rank by soft thresholding k.
        B = low_rankify( B, reg )

        # Proximal step of constraining to bounded norm
        #B = finite_norm( B )

        residuals.append( residual( y, X, B_ ) )
        gradient_norms.append( norm( dB ) )
        print i, residuals[-1], gradient_norms[-1], norm( B - B_ ) / norm( B )

        # Check convergence
        if norm( B - B_ ) / norm( B ) < 1e-5:
            break
        B_ = B
    # Plot graphs
    # plt.plot( residuals )
    # plt.plot( gradient_norms )
    # plt.show()

    return B

def test_solve_exact(iters):
    N = 100
    d = 3
    k = 2

    pi = array( [0.5, 0.5] ) 
    B = eye( d, k )
    #B = randn( d, k )
    B2 = B.dot( diag( pi ) ).dot( B.T )
    X = randn( N, d )
    y = array( [ x.T.dot( B2 ).dot( x ) for x in X ] )

    #B20 = B2 + 0.1 * randn( d, d )
    #B2_ = solve( y, X, B20 )
    B2_ = solve( y, X, iters = iters )

    print residual( y, X, B2 )
    print residual( y, X, B2_ )

    print B2_, B2
    print svdvals(B2_), svdvals(B2)
    print norm(B2 - B2_)/norm(B2)

def test_solve_samples(iters):
    K = 2
    d = 3
    N = 1e3

    fname = "/tmp/test.npz"
    lrm = LinearRegressionsMixture.generate(fname, K, d, cov = "eye", betas="eye")
    M, S, pi, B = lrm.mean, lrm.sigma, lrm.weights, lrm.betas
    B2 = B.dot( diag( pi ) ).dot( B.T )

    y, X = lrm.sample( N )

    B2_ = solve( y, X, B2, iters = iters )

    print residual( y, X, B2 )
    print residual( y, X, B2_ )

    print B2_, B2
    print svdvals(B2_), svdvals(B2)
    print norm(B2 - B2_)/norm(B2)

if __name__ == "__main__":
    test_solve_exact(100)
    test_solve_samples(100)


