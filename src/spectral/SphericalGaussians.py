"""
Spherical Gaussians, from

Hsu and Kakade, "Learning mixtures of spherical Gaussians: moment
methods and spectral decompositions" (2012).
"""

import scipy as sc 
from scipy import diag, array, outer, eye, ones, log, sqrt
from scipy.linalg import norm, svdvals, eig, eigvals, pinv, cholesky
from spectral.linalg import svdk, mrank, approxk, eigen_sep, \
        closest_permuted_matrix, tensorify, matrix_tensorify, \
        column_aerr, column_rerr
from spectral.data import Pairs, TriplesP

from util import DataLogger
from models import GaussianMixtureModel

import time

logger = DataLogger()

def get_whitener( A, k ):
    """Return the matrix W that whitens A, i.e. W^T A W = I. Assumes A
    is k-rank"""

    assert( mrank( A ) == k )
    # Verify PSD
    e = eigvals( A )[:k].real
    if not (e >= 0).all():
      print "Warning: Not PSD"
      print e

    # If A is PSD
    U, S, _ = svdk( A, k )
    A2 = cholesky( U.T.dot( A ).dot( U ) )
    W, Wt = U.dot( pinv( A2 ) ), U.dot( A2 )
    
    return W, Wt

def recover_components( k, P, T, Pe, Te, delta=0.01 ):
    """Recover the k components given input moments M2 and M3 (Pe, Te) are exact P and T"""
    d, _ = P.shape

    # Input error
    logger.add_err( "P", Pe, P, 2 )
    logger.add_consts( "Pe", Pe, k, 2 )
    logger.add_consts( "P", P, k, 2 )
    logger.add_terr( "T", Te, T, d )
    logger.add_tconsts( "Te", Te, d )
    logger.add_tconsts( "T", T, d )

    # Get the whitening matrix of M2
    W, Wt = get_whitener( P, k )
    We, _ = get_whitener( Pe, k )

    logger.add_err( "W", We, W, 2 )
    logger.add_consts( "We", We, k, 2 )
    logger.add_consts( "W", W, k, 2 )

    Tw = lambda theta: W.T.dot( T( W.dot( theta ) ) ).dot( W )
    Twe = lambda theta: We.T.dot( Te( We.dot( theta ) ) ).dot( We )
    #Tw = sc.einsum( 'ijk,ia,jb,kc->abc', T, W, W, W )
    #Twe = sc.einsum( 'ijk,ia,jb,kc->abc', Te, We, We, We )

    logger.add_terr( "Tw", Twe, Tw, k )

    # Repeat [-\log(\delta] times for confidence 1-\delta to find best
    # \theta
    t = int( sc.ceil( -log( delta ) ) )
    best = (-sc.inf, None, None)
    for i in xrange( t ):
        # Project Tw onto a matrix
        theta = sc.rand( k )
        theta = theta/theta.sum()

        # Find the eigen separation
        X = Tw(theta)
        #X = Tw.dot(theta)
        sep = eigen_sep( X )

        if sep > best[0]:
            best = sep, theta, X
    
    # Find the eigenvectors as well
    sep, theta, X = best
    S, U = eig( X, left=True, right=False )
    assert( sc.isreal( S ).all() and sc.isreal( U ).all() )
    S, U = S.real, U.real

    Xe = Twe( theta )
    ##Xe = Twe.dot( theta )
    sepe = eigen_sep( Xe )
    Se, Ue = eig( Xe, left=True, right=False )
    Se, Ue = Se.real, Ue.real

    logger.add( "D", sep/sepe )
    logger.add_err( "lambda", Se, S )
    logger.add_err( "v", Se, S, 'col' )
    
    M = sc.zeros( (d, k) )
    for i in xrange(k):
        M[:, i] = S[i]/theta.dot(U.T[i]) * Wt.dot(U.T[i]) 

    return M

def exact_moments( A, w ):
    """Get the exact moments of a components distribution"""

    k = len(w)
    P = A.dot( diag( w ) ).dot( A.T )
    #T = sum( [ w[i] * tensorify( A.T[i], A.T[i], A.T[i] ) for i in xrange( k ) ] )
    T = lambda theta: A.dot( diag( w) ).dot( diag( A.T.dot( theta ) ) ).dot( A.T )

    return P, T    

def test_exact_recovery():
    """Test the exact recovery of topics"""
    fname = "./test-data/gmm-3-10-0.7.npz"
    gmm = GaussianMixtureModel.from_file( fname )
    k, d, A, w = gmm.k, gmm.d, gmm.means, gmm.weights

    P, T = exact_moments( A, w )

    A_ = recover_components( k, P, T, P, T, delta = 0.01 )
    A_ = closest_permuted_matrix( A.T, A_.T ).T

    print norm( A - A_ )/norm( A )
    print A
    print A_

    assert norm( A - A_ )/norm(A)  < 1e-3

def sample_moments( X, k ):
    """Get the sample moments from data"""
    N, d = X.shape

    # Partition X into two halves to independently estimate M2 and M3
    X1, X2 = X[:N/2], X[N/2:]

    # Get the moments  
    M1 = X1.mean(0)
    M1_ = X2.mean(0)
    M2 = Pairs( X1, X1 ) 
    M3 = lambda theta: TriplesP( X2, X2, X2, theta )
    #M3 = Triples( X2, X2, X2 )

    # Estimate \sigma^2 = k-th eigenvalue of  M2 - mu mu^T
    sigma2 = svdvals( M2 - outer( M1, M1 ) )[k-1]
    assert( sc.isreal( sigma2 ) and sigma2 > 0 )
    # P (M_2) is the best kth rank apprximation to M2 - sigma^2 I
    P = approxk( M2 - sigma2 * eye( d ), k )

    B = matrix_tensorify( eye(d), M1_ )
    T = lambda theta: M3(theta) - sigma2 * ( M1_.dot(theta) * eye( d ) + outer( M1_, theta ) + outer( theta, M1_ ) )
    #T = M3 - sigma2 * ( B + B.swapaxes(2, 1) + B.swapaxes(2, 0) )

    return P, T    

def test_sample_recovery():
    """Test the recovery of topics from samples"""
    fname = "./test-data/gmm-3-10-0.7.npz"
    gmm = GaussianMixtureModel.from_file( fname )
    k, d, A, w = gmm.k, gmm.d, gmm.means, gmm.weights
    X = gmm.sample( 10**5 ) 
    P, T = sample_moments( X, k )

    Pe, Te = exact_moments( A, w )
    del gmm

    A_ = recover_components( k, P, T, Pe, Te )
    A_ = closest_permuted_matrix( A.T, A_.T ).T

    print norm( A - A_ )/norm( A )
    print A
    print A_

    assert norm( A - A_ )/norm( A ) < 5e-1

def main( prefix, N, n, delta, params ):
    """Run on sample in fname"""
    gmm = GaussianMixtureModel.from_file( prefix )
    k, d, M, w = gmm.k, gmm.d, gmm.means, gmm.weights
    logger.add( "M", M )
    logger.add_consts( "M", M, k, 2 )
    logger.add( "w_min", w.min() )
    logger.add( "w_max", w.max() )

    X = gmm.sample( N, n )
    logger.add( "k", k )
    logger.add( "d", d )
    logger.add( "n", n )

    # Set seed for the algorithm
    sc.random.seed( int( params.seed ) )
    logger.add( "seed", int( params.seed ) )

    P, T = sample_moments( X, k )
    Pe, Te = exact_moments( M, w )

    start = time.time()
    M_ = recover_components( k, P, T, Pe, Te, delta = delta )
    stop = time.time()
    logger.add( "time", stop - start )

    M_ = closest_permuted_matrix( M.T, M_.T ).T
    logger.add( "M_", M )

    # Error data
    logger.add_err( "M", M, M_ )
    logger.add_err( "M", M, M_, 'col' )

    print column_aerr(M, M_), column_rerr(M, M_)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "prefix", help="Input file-prefix" )
    parser.add_argument( "ofname", help="Output file (as npz)" )
    parser.add_argument( "--seed", default=time.time(), type=long, help="Seed used" )
    parser.add_argument( "--samples", type=float, help="Number of samples to be used" )
    parser.add_argument( "--subsamples", default=-1, type=float, help="Subset of samples to be used" )
    parser.add_argument( "--delta", default=0.01, type=float, help="Confidence bound" )

    args = parser.parse_args()

    logger = DataLogger(args.ofname)
    main( args.prefix, int(args.samples), int(args.subsamples), args.delta, args )

