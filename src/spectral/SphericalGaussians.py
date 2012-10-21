"""
Spherical Gaussians, from

Hsu and Kakade, "Learning mixtures of spherical Gaussians: moment
methods and spectral decompositions" (2012).
"""

import ipdb
import scipy as sc 
from scipy import diag, array, ndim, outer, eye, ones, log
from scipy.linalg import norm, svd, svdvals, eig, eigvals #, inv, det, cholesky
from spectral.linalg import svdk, mrank, approxk, condition_number, eigen_sep, \
        canonicalise, closest_permuted_matrix, \
        tensor_aerr, tensor_rerr, column_aerr, column_rerr
from spectral.rand import orthogonal
from spectral.data import Pairs, Triples
from generators import GaussianMixtureModel

import time
import logging
logging.basicConfig( level=logging.INFO )

eps = 1e-2

def get_whitener( A, k ):
    """Return the matrix W that whitens A, i.e. W^T A W = I. Assumes A
    is k-rank"""

    assert( mrank( A ) == k )
    # If A is PSD
    U, S, _ = svdk( A, k )
    W, Wt = U.dot( diag( sc.sqrt(S)**-1 ) ), diag(
        sc.sqrt(S) ).dot( U.T )

    # assert( sc.allclose( W.T.dot( A ).dot( W ), sc.eye( k ) ) )
    # assert( sc.allclose( Wt.T.dot( Wt ), A ) )
    
    return W, Wt

def recover_components( P, T, k, Pe, Te, delta=0.01 ):
    """Recover the k components given input moments M2 and M3 (Pe, Te) are exact P and T"""
    d, _ = P.shape

    logging.info( "\|\delta P\|_2\t\t%f\t\t%f", norm(Pe - P, 2), norm(Pe - P, 2)/norm(P, 2) )

    # Consider the k rank approximation of P,
    P = approxk( P, k )
    Pe = approxk( Pe, k )
    logging.info( "\|\delta P_k\|_2\t\t%f\t\t%f", norm(Pe - P, 2), norm(Pe - P, 2)/norm(P, 2) )
    logging.info( "\|P\|_2, \sigma_k(P), K(P)\t\t%f\t\t%f\t\t%f", norm( Pe , 2 ), svdvals(Pe)[k-1], condition_number( Pe, k ) )
    logging.info( "\|P'\|_2, \sigma_k(P'), K(P')\t\t%f\t\t%f\t\t%f", norm( P , 2 ), svdvals(P)[k-1], condition_number( P, k ) )

    # Get the whitening matrix of M2
    W, Wt = get_whitener( P, k )
    We, Wte = get_whitener( Pe, k )
    logging.info( "\|\delta W\|_2\t\t%f\t\t%f", norm(We - W, 2), norm(We - W, 2)/norm(W, 2) )
    logging.info( "\|W\|_2, \sigma_k(W), K(W)\t\t%f\t\t%f\t\t%f", norm( We , 2 ), svdvals(We)[k-1], condition_number( We, k ) )
    logging.info( "\|W'\|_2, \sigma_k(W'), K(W')\t\t%f\t\t%f\t\t%f", norm( W , 2 ), svdvals(W)[k-1], condition_number( W, k ) )

    # Whiten the third moment
    logging.info( "\|\delta T\|_2\t\t%f\t\t%f", tensor_aerr( Te, T, d, 2 ), tensor_rerr( Te, T, d, 2 ) ) 

    Tw = lambda theta: W.T.dot( T( W.dot( theta ) ) ).dot( W )
    Twe = lambda theta: We.T.dot( Te( We.dot( theta ) ) ).dot( We )

    logging.info( "\|\delta T_W\|_2\t\t%f\t\t%f", tensor_aerr( Twe, Tw, k, 2 ), tensor_rerr( Twe, Tw, k, 2 ) ) 

    # Repeat [-\log(\delta] times for confidence 1-\delta to find best
    # \theta
    t = int( sc.ceil( -log( delta ) ) )
    best = (-sc.inf, None, None)
    for i in xrange( t ):
        # Project Tw onto a matrix
        theta = orthogonal( k ).T[0] 

        # Try with eigen vectors
        X = Tw(theta)
        sep = eigen_sep( X )

        if sep > best[0]:
            best = sep, theta, X
    
    # Find the eigenvectors as well
    sep, theta, X
    S, U = eig( X, left=True, right=False )
    assert( sc.isreal( S ).all() )
    assert( sc.isreal( U ).all() )
    S, U = S.real, U.real

    Xe = Twe( theta )
    sepe = eigen_sep( Xe )
    Se, Ue = eig( Xe, left=True, right=False )

    logging.info( "\|\delta \Delta\| \t\t%f\t\t%f", abs( sepe - sep ), abs( sepe - sep )/abs( sep ) )
    logging.info( "\|\delta \lambda_i\| \t\t%f\t\t%f", norm( Se - S ), norm( Se - S )/norm( S ) )
    logging.info( "\|\delta v_i\| \t\t%f\t\t%f", column_aerr( Ue, U ), column_rerr( Ue, U ) ) 
    
    M = []
    for (l, v) in zip( S,  U ):
        m = l/(theta.dot(v)) * Wt.T.dot( v )
        M.append( m )
    M = sc.column_stack( M )

    return M

def exact_moments( A, w ):
    """Get the exact moments of a components distribution"""

    P = A.dot( diag( w ) ).dot( A.T )
    T = lambda theta: A.dot( diag( w ) ).dot( diag( A.T.dot( theta ) ) ).dot( A.T )

    return P, T    

def test_exact_recovery():
    """Test the exact recovery of topics"""

    k = 2 
    d = 3

    w = array( ones( 2 )/2 )
    A = array( [[-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]] ).T
    sigma2 = [0.2 * eye( 3 )] * 2
    gmm = GaussianMixtureModel( w, A, sigma2 )

    P, T = exact_moments( A, w )

    A_ = recover_components( P, T, k, P, T, delta = 0.01 )
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
    M1_ = X2.mean()
    M2 = Pairs( X1, X1 ) 
    M3 = Triples( X2, X2, X2 )

    # Estimate \sigma^2 = k-th eigenvalue of  M2 - mu mu^T
    sigma2 = svdvals( M2 - outer( M1, M1 ) )[k]
    assert( sc.isreal( sigma2 ) and sigma2 > 0 )
    # P (M_2) is the best kth rank apprximation to M2 - sigma^2 I
    P = approxk( M2 - sigma2 * eye( d ), k )

    T = lambda theta: M3(theta) - sigma2 * ( theta.dot(M1_) * eye( d ) +
            outer( theta, M1_ ) + outer( M1_, theta ) )

    return P, T    

def test_sample_recovery():
    """Test the recovery of topics from samples"""

    k = 2 
    d = 3

    # Generate data from the LDA model
    w = array( ones( 2 )/2 )
    A = array( [[-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]] ).T
    sigma2 = [0.2 * eye( 3 )] * 2
    gmm = GaussianMixtureModel( w, A, sigma2 )

    X = gmm.sample( 100000 ) # Normalising for the words
    P, T = sample_moments( X, k )
    Pe, Te = exact_moments( A, w )

    A_ = recover_components( P, T, k, Pe=Pe, Te=Te )
    A_ = closest_permuted_matrix( A.T, A_.T ).T

    print norm( A - A_ )/norm( A )
    print A
    print A_

    assert norm( A - A_ )/norm( A ) < 1e-3

def main( fname, samples, delta ):
    """Run on sample in fname"""

    gmm = sc.load( fname )
    k, d, M, w, X = gmm['k'], gmm['d'], gmm['M'], gmm['w'], gmm['X']
    logging.info( "\|M\|_2, \sigma_k(M), K(M)\t\t%f\t\t%f\t\t%f", norm( M , 2 ), svdvals(M)[k-1], condition_number( M, k ) )
    logging.info( "w_min, w_max\t\t%f\t\t%f", w.min(), w.max() )

    N, _ = X.shape
    if (samples < 0 or samples > N):
        print "Warning: %s greater than number of samples in file. Using %s instead." % ( samples, N )
    else:
        X = X[:samples, :]

    P, T = sample_moments( X, k )
    Pe, Te = exact_moments( M, w )

    start = time.time()
    M_ = recover_components( P, T, k, delta = delta, Pe = Pe, Te = Te )
    stop = time.time()
    M_ = closest_permuted_matrix( M.T, M_.T ).T

    # Error data
    logging.info( "\|\delta M\|_F\t\t%f\t\t%f", norm(M - M_), norm(M - M_)/norm(M) )
    logging.info( "\|\delta mu_i\|_2\t\t%f\t\t%f", column_aerr( M, M_ ), column_rerr( M, M_ ) )
    logging.info( "Running Time: %f", stop - start )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "fname", help="Input file (as npz)" )
    parser.add_argument( "--seed", default=time.time(), type=long, help="Seed used" )
    parser.add_argument( "--samples", default=-1, type=float, help="Number of samples to be used" )
    parser.add_argument( "--delta", default=0.01, type=float, help="Confidence bound" )

    args = parser.parse_args()
    print "Seed:", int( args.seed )
    sc.random.seed( int( args.seed ) )

    main( args.fname, int(args.samples), args.delta )

