"""
Quick implementation of the mixture of linear regressions code
"""

import shutil, tempfile 
import ipdb

import scipy as sc
import scipy.spatial
import scipy.linalg
from scipy import diag, array, ndim, outer, eye, ones,\
        log, sqrt, zeros, floor, exp
from scipy.linalg import norm, svd, svdvals, eig, eigvals, inv, pinv, cholesky, solve
from scipy.spatial.distance import cdist

permutation, rand, dirichlet = scipy.random.permutation, scipy.random.rand, scipy.random.dirichlet

from spectral.linalg import tensorify, closest_permuted_matrix, \
        mrank, condition_number 
from spectral.data import Pairs, Pairs2

from models import LinearRegressionsMixture 

from spectral.MultiView import recover_M3

from optim import PhaseRecovery, TensorRecovery

def recover_B( k, y, X, iters = 50, alpha0 = 0.1, reg = 0, B20B30 = (None,None), B2B3 = (None, None) ):
    """Recover the mixture weights B"""
    B20, B30 = B20B30
    B2, B3 = B2B3

    # Use convex optimisation to recover B2 and B3
    B2_ = PhaseRecovery().solve( y**2, X, B0 = B20, alpha = "1/sqrt(T)", alpha0 = alpha0, iters = iters, reg = reg, verbose = False )
    print norm( B20 - B2 ), norm( B2_ - B2 ) 

    B3_ = TensorRecovery().solve( y**3, X, B0 = B30, alpha = "1/sqrt(T)", alpha0 = alpha0, iters = iters, reg = reg, verbose = False )
    print norm( B30 - B3 ), norm( B3_ - B3 ) 

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

def smooth_data( y, X, smoothing, smoothing_dimensions = None):
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
        return y, X
    elif smoothing == "all":
        Q = ones( N )/N
        y, X = Q.dot( y ), Q.dot( X )
        return sc.atleast_1d( y ), sc.atleast_2d( X )
    elif smoothing == "local":
        # Choose smoothing_dimensions number of random Xs
        Zi = permutation(N)[:smoothing_dimensions]
        Z = X[ Zi ]
        Q = exp( - cdist( Z, X )**2 )
        # Normalise to be stochastic
        Q = (Q.T/Q.sum(1)).T

        y, X = Q.dot( y ), Q.dot( X )
        return y, X
    elif smoothing == "subset":
        # Choose smoothing_dimensions number of random Xs
        Q = zeros( (smoothing_dimensions, N) )
        for i in xrange( smoothing_dimensions ):
            Zi = permutation(N)[:smoothing_dimensions]
            Q[ i, Zi ] = 1.0/len(Zi)

        y, X = Q.dot( y ), Q.dot( X )
        return y, X
    elif smoothing == "dirichlet":
        # Choose smoothing_dimensions number of random Xs
        alpha = 0.1
        Q = dirichlet( alpha * ones(N)/N, smoothing_dimensions )

        y, X = Q.dot( y ), Q.dot( X )
        return y, X
    elif smoothing == "white":
        # Choose smoothing_dimensions number of random Xs
        Q = pinv( X )

        y, X = Q.dot( y ), Q.dot( X )
        return y, X
    elif smoothing == "random":
        # Choose smoothing_dimensions number of random Xs
        Q = rand(smoothing_dimensions, N)
        # Normalise to be stochastic
        Q = (Q.T/Q.sum(1)).T

        y, X = Q.dot( y ), Q.dot( X )
        return y, X
    else: 
        raise NotImplementedError()

def recover_B2_ridge( y, X, reg = 0 ):
    """Recover B2 using ridge regression"""
    N, D = X.shape

    y = y**2
    indices = sc.tril_indices(D)

    X = array( [ (outer(x,x)[indices]) for x in X ] )
    B2_ = inv(X.T.dot(X) + reg * eye(X.shape[1])).dot( X.T ).dot( y )

    B2 = zeros((D,D))

    B2[indices] = B2_
    B2 = (B2 + B2.T)/2

    return B2

def recover_B3_ridge( y, X, reg = 0 ):
    """Recover B2 using ridge regression"""
    N, D = X.shape

    y = y**3
    indices = []
    multiplicity = []
    for i in xrange(D):
        for j in xrange(i+1):
            for k in xrange(j+1):
                indices.append( [i, j, k] )
                multiplicity.append( 1 if i == j == k else (3 if (i == j or j == k or k == i ) else 6)) 
    indices = zip(*indices)

    X = array( [ (tensorify(x,x,x)[indices]) / multiplicity for x in X ] )
    B3_ = inv(X.T.dot(X) + reg * eye(X.shape[1])).dot( X.T ).dot( y )

    B3d = zeros((D,D,D))
    B3d[indices] = B3_
    B3 = zeros((D,D,D))

    for i in xrange(D):
        for j in xrange(D):
            for k in xrange(D):
                idx = [i,j,k]
                idx.sort()
                B3[i,j,k]  = B3d[idx[0], idx[1], idx[2]]

    return B3

def main( args ):
    K, d, N = args.k, args.d, int( args.samples )

    # Initialise the model
    if args.model is not None:
        lrm = LinearRegressionsMixture.from_file( args.model )
    else:
        sc.random.seed(args.seed)
        fname = tempfile.mktemp()
        lrm = LinearRegressionsMixture.generate(fname, K, d, weights = "uniform", cov = "eye", betas="eye")
    # Generate some samples
    y, X = lrm.sample( N )

    # Compute exact moments
    _, _, pi, B = lrm.mean, lrm.sigma, lrm.weights, lrm.betas
    B2 = B.dot( diag( pi ) ).dot( B.T )
    B3 = sum( [ pi[i] * tensorify(B.T[i], B.T[i], B.T[i] ) for i in xrange(K) ] )

    # Add some noise to y
    if args.with_noise:
        sigma2 = 0.2
        noise = sc.randn(*y.shape) * sqrt( sigma2 )
        y += noise
    sc.random.seed(args.seed)

    Q2 = None
    y, X = smooth_data( y, X, args.smoothing, args.smoothing_dimensions )

    if args.init == "zero":
        B20 = sc.zeros( B2.shape )
        B30 = sc.zeros( B3.shape )
    elif args.init == "random":
        B20 = sc.randn(*B2.shape)
        B30 = sc.randn(*B3.shape)
    elif args.init == "near-optimal":
        B20 = B2 + 0.1 * sc.randn(*B2.shape)
        B30 = B3 + 0.1 * sc.randn(*B3.shape)
    elif args.init == "ridge":
        B20 = recover_B2_ridge( y, X, args.rreg )
        B30 = recover_B3_ridge( y, X, args.rreg )
    else:
        raise NotImplementedError()
    
    B_, B2_, B3_ = recover_B( K, y, X, int( args.iters ), float(args.alpha0), float(args.reg), (B20, B30), (B2, B3) )
    B_ = closest_permuted_matrix( B.T, B_.T ).T
    print norm( B - B_ )

    del lrm

if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument( "-k", type=int, help="number of clusters" )
    parser.add_argument( "-d", type=int, help="number of dimensions" )
    parser.add_argument( "--model", default=None, type=str, help="Load model from" )
    parser.add_argument( "--seed", default=int(time.time() * 100), type=int,
            help="Seed used for algorithm (separate from generation)" )
    parser.add_argument( "--samples", default=1e4, type=float, help="Number of samples to be used" )
    parser.add_argument( "--init", default="zero", type=str, help="How to initialise" )
    parser.add_argument( "--alpha0", default=1e-1, type=float, help="Starting pace for gradient descent" )
    parser.add_argument( "--reg", default=0, type=float, help="Regularization for SVD" )
    parser.add_argument( "--rreg", default=0, type=float, help="Regularization for ridge" )
    parser.add_argument( "--iters", default=1e2, type=float, help="Number of iterations of gradient descent" )
    parser.add_argument( "--with-noise", default=False, type=bool, help="Use noise" )
    parser.add_argument( "--smoothing", default="none", type=str, help="Smoothing scheme used; eye | all | local | random" )
    parser.add_argument( "--smoothing-dimensions", default=None, type=int, help="Number of dimensions after smoothing" )

    args = parser.parse_args()
    main( args )

