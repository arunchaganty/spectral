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
from spectral.rand import orthogonal, wishart
from spectral.data import Pairs, TriplesP, PairsQ, TriplesPQ
from models import LinearRegressionsMixture 

def recover_parameters( k, d, M2, M3 ):
    """M2 is a vector, while M3 is a tensor"""
    # Extract B2 by projecting on a large number of S 
    
    B2_ = zeros( d * (d+1) / 2 )
    B2__ = zeros( d * (d+1) / 2 )
    Ss = wishart( d, eye(d), d * (d+1) / 2 )

    for i in xrange( d * (d+1) / 2 ):
        B2_[i] = M2( Ss[i] )
        B2__[i] = M2( Ss[i] )/2
    # Get the projection 
    Ss = array( [ S_[sc.triu_indices( d )] for S_ in Ss ] )
    Ss_inv = inv(Ss.T.dot(Ss)).dot( Ss.T )
    B2_ = Ss_inv.dot( B2_ )
    B2__ = Ss_inv.dot( B2__ )

    # Symmetrise
    B2 = zeros( (d ,d) )
    B2[ sc.triu_indices( d ) ] = B2__
    B2[ sc.tril_indices(d) ] = B2.T[ sc.tril_indices(d) ]

    idx = zip(* sc.triu_indices( d ) )
    diag_idx = [i for i in xrange( len(idx) ) if idx[i][0] == idx[i][1]]
    for i in xrange(d) :
        B2[ i, i ] = B2_[diag_idx[i]]

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

    B2_ = recover_parameters( k, d, M2, M3 )

    ipdb.set_trace()

    assert( sc.allclose( B2, B2_ ) )

