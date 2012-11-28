"""
Quick implementation of the mixture of linear regressions code
"""

import ipdb
import scipy as sc 
from scipy import diag, array, ndim, outer, eye, ones, log, sqrt, zeros, floor
from scipy.linalg import norm, svd, svdvals, eig, eigvals, inv, det, cholesky
from spectral.linalg import svdk, mrank, approxk, eigen_sep, \
        closest_permuted_matrix, tensorify, matrix_tensorify, \
        column_aerr, column_rerr,\
        condition_number, column_gap, column_sep
from spectral.rand import orthogonal, wishart, dirichlet
from spectral.data import Pairs, Triples, PairsQ, TriplesQ
from models import LinearRegressionsMixture 

def recover_parameters( k, d, N, X2, X3, Y2, Y3 ):
    """X2: vector -> matrix, while X3: vector -> tensor"""
    # Extract B2 by projecting on a large number of S 
    B2 = zeros( d * (d+1) / 2 )

    # The transform
    T2 = zeros( (d * (d+1)/2, d * (d+1)/2) )

    # Generate the qs
    Q = dirichlet( 0.1 * ones(N), d**2 )

    B2_ = zeros( d*(d+1)/2 )

    for i in xrange( d * (d+1)/2 ):
        q = Q[i]
        T2[i] = X2(q)[sc.triu_indices( d )]
        B2_[i] = Y2(q)
    ipdb.set_trace()

    # Get the pseudo inverse 
    T2_inv = inv( T2.T.dot(T2) ).dot( T2.T )

    # Symmetrise
    B2 = zeros( (d ,d) )
    B2[ sc.triu_indices( d ) ] = T2_inv.dot(B2_/2)
    B2[ sc.tril_indices(d) ] = B2.T[ sc.tril_indices(d) ]

    idx = zip(* sc.triu_indices( d ) )
    tdiag_idx = [d*i - i*(i-1)/2 for i in xrange( d ) ]
    diagB2 = T2_inv.dot(B2_)[tdiag_idx]
    for i in xrange( d ):
        B2[i,i] = diagB2[i]

    return B2

def exact_moments( k, pi, beta, mu, S ):
    """Get the exact moments from the model parameters"""
    (d, k) = beta.shape

    B2 = beta.dot( diag( pi ) ).dot( beta.T )
    B3 = lambda theta: beta.dot( diag( pi ).diag( beta.T.dot( theta ) ) ).dot( beta.T )

    # Get the square root of W
    W = cholesky( S )

    def M2( S_ ):
        #S_ = W.T.dot( S ).dot( W )
        return S_.flatten().dot( B2.flatten() )

    def M3( S_, theta ):
        #S_ = W.T.dot( S ).dot( W )
        return S_.flatten().dot( B3(theta).flatten() )

    return M2, M3, B2

def test_exact_recovery():
    """Test the accuracy of exact recovery"""
    k = 3
    d = 10

    fname = "/tmp/test.npz"
    lrm = LinearRegressionsMixture.generate(fname, k, d)
    M, S, pi, B = lrm.mean, lrm.sigma, lrm.weights, lrm.betas

    M2, M3, B2 = exact_moments( k, pi, B, M, S )

    #B2_ = recover_parameters( k, d, M2, M3 )

    #ipdb.set_trace()

    #assert( sc.allclose( B2, B2_ ) )

def normalise( y, X ):

    # Normalise data 
    # Center
    mu = X.mean(0)
    X = X - mu
    # Whiten
    S = Pairs( X, X )
    W = cholesky( S )
    X = X.dot( inv( W ) )

    return y, X, mu, S

def sample_moments( k, y, X ):
    """Get the sample moments"""
    (N, d) = X.shape

    # Normalise data 
    # Center
    #mu = X.mean(0)
    #X = X - mu
    ## Whiten
    #S = Pairs( X, X )
    #W = cholesky( S )
    #X = X.dot( inv( W ) )

    def X2( q ):
        return PairsQ( X, q )
    
    def Y2( q ):
        return (y**2).dot( q )/N
    
    def X3( q ):
        return TriplesQ( X, q )

    def Y3( q ):
        return (y**3).dot( q )/N

    return X2, X3, Y2, Y3

def test_sample_recovery():
    """Test the accuracy of sample recovery"""
    k = 3
    d = 3
    N = 1e5

    fname = "/tmp/test.npz"
    lrm = LinearRegressionsMixture.generate(fname, k, d, cov = "eye")
    M, S, pi, B = lrm.mean, lrm.sigma, lrm.weights, lrm.betas

    y, X = lrm.sample( N )

    B2 = B.dot( diag( pi ) ).dot( B.T )

    X2, X3, Y2, Y3 = sample_moments( k, y, X )

    B2_ = recover_parameters( k, d, N, X2, X3, Y2, Y3 )
    #B2_ = closest_permuted_matrix( B2.T, B2_.T ).T

    print ( norm( B2 - B2_ ) )

    ipdb.set_trace()

    assert( sc.allclose( B2, B2_ ) )

