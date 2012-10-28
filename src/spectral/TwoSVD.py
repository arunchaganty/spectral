"""
The Excess Correlation Analysis method from 

Anandkumar, Foster, Hsu, et. al, "Two SVDs Suffioce: Spectral
decompositions for probabilistic topic modelling and latent Dirichlet
allocation" (2012).
"""

#import ipdb
import scipy as sc 
from scipy import diag, array, ndim, outer
from scipy.linalg import norm, svd #, inv, det, cholesky
from spectral.linalg import svdk, mrank, approxk, \
        canonicalise, closest_permuted_matrix
from spectral.rand import orthogonal
from spectral.data import Pairs, Triples
from models import LDATopicModel

eps = 1e-2

def get_whitener( A, k ):
    """Return the matrix W that whitens A, i.e. W^T A W = I. Assumes A
    is k-rank"""

    assert( mrank( A ) == k )
    # If A is PSD
    U, S, _ = svdk( A, k )
    W, Wt = U.dot( diag( sc.sqrt(S)**-1 ) ) , ( diag( sc.sqrt(S) ) ).dot( U.T )

    # assert( sc.allclose( W.T.dot( A ).dot( W ), sc.eye( k ) ) )
    # assert( sc.allclose( Wt.T.dot( Wt ), A ) )
    
    return W, Wt

# LDA 
def recover_topics( P, T, k, a0 ):
    """Recover the k components given input Pairs and Triples and
    $\\alpha_0$"""

    # Consider the k rank approximation of P,
    P = approxk( P, k )
    # Get the whitening matrix and coloring matrices
    W, Wt = get_whitener( P, k )

    # Whiten the third moment
    Tw = lambda theta: W.T.dot( T( W.dot(theta) ) ).dot( W )

    # Project Tw onto a matrix
    theta = orthogonal( k ).T[0] 

    U, S, _ = svd( Tw( theta ) )
    assert( (S > 1e-10).all() ) # Make sure it is non-singular

    O = []
    for i in xrange( k ):
        v = U.T[i]
        Zinv = (a0 + 2)/2 * (v.T.dot(Tw(v)).dot(v))
        O.append( Zinv * Wt.T.dot( v ) )

    O = sc.column_stack( O )

    return abs( O )

def exact_moments( alphas, topics ):
    """Get the exact moments of a components distribution"""

    a0 = alphas.sum()
    O = topics

    P = 1/((a0 +1)*a0) * O.dot( diag( alphas ) ).dot( O.T )
    T = lambda theta: 2/((a0+2)*(a0 +1)*a0) * O.dot( diag( O.T.dot(
        theta ) ) ).dot( diag( alphas ) ).dot( O.T )

    return P, T    

def test_exact_recovery():
    """Test the exact recovery of topics"""

    k = 3 
    d = 100

    # Generate data from the LDA model
    lda = LDATopicModel.generate( k, d )
    O = lda.topics 
    a0 = lda.alphas.sum()

    P, T = exact_moments( lda.alphas, O)

    O_ = recover_topics( P, T, k, a0 )

    O_ = closest_permuted_matrix( O.T, O_.T ).T

    assert norm( O - O_ ) < 1e-3

def sample_moments( X1, X2, X3, k, a0 ):
    """Get the sample moments from data
    Assumes every row of the document corresponds to one data point
    """
    #N, W = X1.shape

    # Get three uncorrelated sections of the data
    M1 = X1.mean(0)
    M2 = Pairs( X1, X2 )
    M3 = Triples( X1, X2, X3 )

    P = M2 - a0/(a0+1) * outer(M1, M1)
    T = lambda theta: (M3(theta) - a0/(a0+2) * (M2.dot( outer(theta, M1))
            + outer( M1, theta ).dot( M2 ) + theta.dot(M1) * M2 )  +
        2 * a0**2/((a0+2)*(a0+1)) * (theta.dot(M1) * outer(M1, M1)))

    return P, T    

def test_sample_recovery():
    """Test the exact recovery of topics"""

    k = 3 
    d = 10

    # Generate data from the LDA model
    lda = LDATopicModel.generate( k, d )
    a0 = lda.alphas.sum()
    O = lda.topics 

    X1, X2, X3 = lda.sample( 1000, words=1000 ) # Normalising for the words
    P, T = sample_moments( X1, X2, X3, k, a0 )

    O_ = recover_topics( P, T, k, a0 )
    
    O_ = closest_permuted_matrix( O, O_ )

    print O
    print O_

    assert norm( O - O_ ) < 1e-3

def main( fname ):
    """Run on sample in fname"""

    lda = sc.load( fname )
    k, d, a0, O, X = lda['k'], lda['d'], lda['a0'], lda['O'], lda['data']
    X1, X2, X3 = X

    P, T = sample_moments( X1, X2, X3, k, a0 )

    O_ = recover_topics( P, T, k, a0 )
    O_ = closest_permuted_matrix( O.T, O_.T ).T

    print k, d, a0, norm( O - O_ )

    #print O
    #print O_

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "fname", help="Input file (as npz)" )

    args = parser.parse_args()

    main( args.fname )

