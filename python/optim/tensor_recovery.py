"""
Phase recovery of a tensor using proximal subgradient descent
"""

import ipdb
import scipy as sc
from scipy import diag, array, ndim, outer, eye, ones, log, sqrt, zeros, floor, exp, einsum
from scipy.linalg import norm, svd, svdvals, eig, eigvals, inv, det, cholesky
rand, randn = sc.rand, sc.randn

from models import LinearRegressionsMixture
from spectral.data import Pairs
import spectral.linalg as sl
from spectral.linalg import tensorify, HOSVD

#import matplotlib.pyplot as plt

def gradient_step( y, X, B ):
    """
    Take a gradient step along each B to optimise 
    $\diff{L}{B_{ij}} = \sum_i (x^i' B x^i - y^i^2) x^i_i x^i_j

    Including the constraint that $B$ is symmetric, this becomes,
    $\diff{L}{B_{ij}} = 2 \sum_i (x^i' B x^i - y^i^2) x^i_i x^i_j
    """ 

    d = B.shape[0]
    N = len(y)

    # Compute x^T B x - y
    dB = zeros( (d, d, d) )
    for (y_i, x_i) in zip( y, X ):
        dB += (einsum("ijk,i,j,k", B, x_i, x_i, x_i)  - y_i) * tensorify( x_i, x_i, x_i )

    return dB/N

def low_rankify( X, thresh ):
    """Make X low rank"""
    shape = X.shape
    # Threshold the singular values of each mode of the tensor
    for i in xrange( X.ndim ):
        X_i = sl.unfold( X, i+1 )
        U, S, Vt = svd( X_i, full_matrices = False )
        S -= thresh
        S[ S < 0.0 ] = 0
        X = sl.fold( U.dot( diag( S ) ).dot( Vt ), i+1, X.shape )
    return X

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
        tot += (einsum( "ijk,i,j,k", B, x_i, x_i, x_i) - y_i)**2
    return tot/N

def solve( y, X, B0 = None, reg = 0, iters = 500, alpha = "1/T" ):
    """
    Solve for B that optimises the following objective:
    $\\argmin_{B} = 1/2 * \sum_i (x_i^T B x_i - y_i^2)^2/N + \lambda Tr(B)$
    Every row of X corresponds to a y
    """

    N, d = X.shape

    # Precompute X X^T

    # Initialise the B0
    if B0 is None:
        B0 = zeros( (d, d, d) ) 

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

def test_solve_exact(samples, iters):
    K = 2
    d = 3
    N = samples

    pi = array( [0.5, 0.5] ) 
    B = eye( d, K )
    #B = randn( d, k )
    B3 = sum( [ pi[i] * tensorify(B.T[i], B.T[i], B.T[i] ) for i in xrange(K) ] )
    X = randn( N, d )
    y = array( [ einsum( "ijk,i,j,k", B3, x, x, x ) for x in X ] )

    B3_ = solve( y, X, iters = iters )

    print residual( y, X, B3 )
    print residual( y, X, B3_ )

    print B3_, B3
    _, _, V = HOSVD( B3 )
    __, _, V_ = HOSVD( B3_ )
    print V, V_
    print norm(B3 - B3_)/norm(B3)

def test_solve_samples(samples, iters):
    K = 2
    d = 3
    N = samples

    fname = "/tmp/test.npz"
    lrm = LinearRegressionsMixture.generate(fname, K, d, cov = "eye", betas="random")
    M, S, pi, B = lrm.mean, lrm.sigma, lrm.weights, lrm.betas
    B3 = sum( [ pi[i] * tensorify(B.T[i], B.T[i], B.T[i] ) for i in xrange(K) ] )

    y, X = lrm.sample( N )

    B3_ = solve( y**3, X, B3, iters = iters )

    print B3_, B3
    print norm(B3 - B3_)/norm(B3)

if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument( "--samples", default=1e6, type=float, help="Number of samples to be used" )
    parser.add_argument( "--iters", default=1e2, type=float, help="Number of iterations to be used" )
    parser.add_argument( "--with-noise", default=False, type=bool, help="Use noise" )

    args = parser.parse_args()

    #test_solve_exact(int(args.samples), int( args.iters) )
    test_solve_samples(int(args.samples), int( args.iters) )


