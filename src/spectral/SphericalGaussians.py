"""
Spherical Gaussians, from

Hsu and Kakade, "Learning mixtures of spherical Gaussians: moment
methods and spectral decompositions" (2012).
"""

import ipdb
import scipy as sc 
from scipy import diag, array, ndim, outer, eye, ones, log
from scipy.linalg import norm, svd, svdvals, eig #, inv, det, cholesky
from spectral.linalg import svdk, mrank, approxk, \
        canonicalise, closest_permuted_matrix
from spectral.rand import orthogonal
from spectral.data import Pairs, Triples
from generators import GaussianMixtureModel

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

def recover_components( P, T, k, delta=0.01 ):
    """Recover the k components given input moments M2 and M3"""

    # Consider the k rank approximation of P,
    P = approxk( P, k )

    # Get the whitening matrix of M2
    W, Wt = get_whitener( P, k )

    # Whiten the third moment
    Tw = lambda theta: W.T.dot( T( W.dot( theta ) ) ).dot( W )


    # Repeat [-\log(\delta] times for confidence 1-\delta
    t = int( sc.ceil( -log( delta ) ) )
    best = (-sc.inf, None, None)
    for i in xrange( t ):
        # Project Tw onto a matrix
        theta = orthogonal( k ).T[0] 

        # Try with eigen vectors
        S, U = eig( Tw(theta), left=True, right=False )
        assert( sc.isreal( S ).all() )
        assert( sc.isreal( U ).all() )
        S, U = S.real, U.real

        lv = zip( S, U.T )
        lv.sort()
        l = array( map( lambda x: x[0], lv ) )

        # Compare min of |\lambda_i| and |\lambda_i - \lambda_j|
        diff = min( abs(l).min(), abs(sc.diff( l )).min() )
        if diff > best[0]:
            best = diff, theta, lv

        M = []
        for (l, v) in lv:
            m = l/(theta.dot(v)) * Wt.T.dot( v )
            M.append( m )
        M = sc.column_stack( M )
        print diff, M


    diff, theta, lv = best
    M = []
    for (l, v) in lv:
        m = l/(theta.dot(v)) * Wt.T.dot( v )
        M.append( m )
    M = sc.column_stack( M )
    print "Final: ", diff, M

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

    A_ = recover_components( P, T, k, delta = 0.1 )
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

    A_ = recover_components( P, T, k )
    A_ = closest_permuted_matrix( A.T, A_.T ).T

    print norm( A - A_ )/norm( A )
    print A
    print A_

    assert norm( A - A_ )/norm( A ) < 1e-3

# def main( fname ):
#     """Run on sample in fname"""
# 
#     lda = sc.load( fname )
#     k, d, a0, O, X = lda['k'], lda['d'], lda['a0'], lda['O'], lda['data']
#     X1, X2, X3 = X
# 
#     P, T = sample_moments( X1, X2, X3, k, a0 )
# 
#     O_ = recover_topics( P, T, k, a0 )
#     O_ = closest_permuted_matrix( O.T, O_.T ).T
# 
#     print k, d, a0, norm( O - O_ )
# 
#     #print O
#     #print O_
# 
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument( "fname", help="Input file (as npz)" )
# 
#     args = parser.parse_args()
# 
#     main( args.fname )
# 
