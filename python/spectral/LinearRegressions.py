"""
Quick implementation of the mixture of linear regressions code
"""

import shutil, tempfile 

import scipy as sc
import scipy.spatial
import scipy.linalg
from scipy import diag, array, ndim, outer, eye, ones,\
        log, sqrt, zeros, floor, exp
from scipy.linalg import norm, svd, svdvals, eig, eigvals, inv 
from scipy.spatial.distance import cdist

permutation, rand, dirichlet = scipy.random.permutation, scipy.random.rand, scipy.random.dirichlet

from spectral.linalg import tensorify, closest_permuted_matrix 

from models import LinearRegressionsMixture 

from spectral.MultiView import recover_M3

from optim import PhaseRecovery, TensorRecovery

def recover_B( k, y, X, iters = 50, Q = None ):
    """Recover the mixture weights B"""

    # Use convex optimisation to recover B2 and B3
    B2 = PhaseRecovery().solve( y**2, X, Q, alpha0 = 0.1, iters = iters, verbose = False )
    B3 = TensorRecovery().solve( y**3, X, Q, alpha0 = 0.01, iters = iters, reg = 1e-3, verbose = False )

    B3_ = lambda theta: sc.einsum( 'abj,j->ab', B3, theta )

    B = recover_M3( k, B2, B2, B3_ )

    return B, B2, B3

def test_sample_recovery():
    """Test the accuracy of sample recovery"""
    K = 3
    d = 3
    N = 1e5

    fname = "/tmp/test.npz"
    lrm = LinearRegressionsMixture.generate(fname, K, d, cov = "eye", 
            betas="eye")
    _, _, pi, B = lrm.mean, lrm.sigma, lrm.weights, lrm.betas

    # Compute exact moments
    B2 = B.dot( diag( pi ) ).dot( B.T )
    B3 = sum( [ pi[i] * tensorify(B.T[i], B.T[i], B.T[i] ) 
        for i in xrange(K) ] )

    # Generate some samples
    y, X = lrm.sample( N )

    # Add some noise to y
    if args.with_noise:
        sigma2 = 0.2
        noise = sc.randn(*y.shape) * sqrt( sigma2 )
        y += noise

    B_, B2_, B3_ = recover_B( K, y, X )
    B_ = closest_permuted_matrix( B.T, B_.T ).T

    print norm( B - B_ ), norm( B2 - B2_ ), norm( B3 - B3_ )

    del lrm

    assert( norm( B - B_) < 1e-1 )

def make_smoothener( y, X, smoothing, smoothing_dimensions = None):
    """
    Make a smoothener based on the scheme:
    none - return eye(N) - no mixing between Xs
    all - return 1_N/N complete mixing between Xs
    local - return cdist( x_m, X_N ) partial local mixing between Xs
    subset - return a partition of m random Xs
    random - return rand( m, N ) partial local mixing between Xs
    """

    N, d = X.shape

    if smoothing_dimensions == None:
        smoothing_dimensions = N

    if smoothing == "none":
        return eye( N )
    elif smoothing == "all":
        return ones( N )/N
    elif smoothing == "local":
        # Choose smoothing_dimensions number of random Xs
        Zi = permutation(N)[:smoothing_dimensions]
        Z = X[ Zi ]
        Q =  cdist( Z, X )
        # Normalise to be stochastic
        Q = (Q.T/Q.sum(1)).T
        return Q
    elif smoothing == "subset":
        # Choose smoothing_dimensions number of random Xs
        Q = zeros( (smoothing_dimensions, N) )
        for i in xrange( smoothing_dimensions ):
            Zi = permutation(N)[:smoothing_dimensions]
            Q[ i, Zi ] = 1.0/len(Zi)
        return Q
    elif smoothing == "dirichlet":
        # Choose smoothing_dimensions number of random Xs
        alpha = 0.1
        Q = dirichlet( alpha * ones(N)/N, smoothing_dimensions )
        return Q
    elif smoothing == "random":
        # Choose smoothing_dimensions number of random Xs
        Q = rand(smoothing_dimensions, N)
        # Normalise to be stochastic
        Q = (Q.T/Q.sum(1)).T
        return Q
    else: 
        raise NotImplementedError()

def main( args ):
    sc.random.seed(args.seed)

    K, d, N = args.k, args.d, int( args.samples )

    # Initialise a model
    fname = tempfile.mktemp()
    lrm = LinearRegressionsMixture.generate(fname, K, d, cov = "eye", betas="eye")
    M, S, pi, B = lrm.mean, lrm.sigma, lrm.weights, lrm.betas

    # Compute exact moments
    B2 = B.dot( diag( pi ) ).dot( B.T )
    B3 = sum( [ pi[i] * tensorify(B.T[i], B.T[i], B.T[i] ) for i in xrange(K) ] )

    # Generate some samples
    y, X = lrm.sample( N )

    Q = make_smoothener( y, X, args.smoothing, args.smoothing_dimensions )

    # Add some noise to y
    if args.with_noise:
        sigma2 = 0.2
        noise = sc.randn(*y.shape) * sqrt( sigma2 )
        y += noise

    B_, B2_, B3_ = recover_B( K, y, X, int( args.iters ) )
    B_ = closest_permuted_matrix( B.T, B_.T ).T

    print norm( B - B_ ),  norm(B2 - B2_), norm(B3 - B3_)

    del lrm

if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument( "-k", type=int, help="number of clusters" )
    parser.add_argument( "-d", type=int, help="number of dimensions" )
    parser.add_argument( "--seed", default=int(time.time() * 100), type=int,
            help="Seed used for algorithm (separate from generation)" )
    parser.add_argument( "--samples", default=1e4, type=float, help="Number of samples to be used" )
    parser.add_argument( "--iters", default=1e2, type=float, help="Number of iterations of gradient descent" )
    parser.add_argument( "--with-noise", default=False, type=bool, help="Use noise" )
    parser.add_argument( "--smoothing", default="none", type=str, help="Smoothing scheme used; eye | all | local | random" )
    parser.add_argument( "--smoothing-dimensions", default=None, type=int, help="Number of dimensions after smoothing" )

    args = parser.parse_args()
    main( args )

