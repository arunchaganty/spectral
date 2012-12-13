"""
Quick implementation of the mixture of linear regressions code
"""

import ipdb
import scipy as sc 
import scipy.spatial
import scipy.linalg
from scipy import diag, array, ndim, outer, eye, ones, log, sqrt, zeros, floor, exp
from scipy.linalg import norm, svd, svdvals, eig, eigvals, inv, det, cholesky
from scipy.spatial.distance import cdist
from spectral.linalg import svdk, mrank, approxk, eigen_sep, \
        closest_permuted_matrix, tensorify, matrix_tensorify, \
        column_aerr, column_rerr,\
        condition_number, column_gap, column_sep
from spectral.rand import orthogonal, wishart, dirichlet
from spectral.data import Pairs, Triples, PairsQ, TriplesQ
from models import LinearRegressionsMixture 

def recover_B2( k, d, N, y, X ):
    """ Extract B2 by projecting onto various q """
    l = 1.0

    indices = sc.triu_indices( d )
    d_ = d * (d+1) / 2

    # Pick a random set of d_ x's from the X
    X0 = X[:d_]
    Q = exp( - cdist( X, X0 )**2 / l )

    # The transform
    Theta = zeros( (d_, d_) )
    for i in xrange( d_ ):
        q = Q.T[i]
        Theta[i,:d_] = PairsQ(X, q)[indices]
    B2_ = (y**2).dot(Q)/N

    B2 = zeros( (d, d) )
    B2[ indices ] = inv(Theta).dot(B2_)
    B2 = (B2 + B2.T)/2

    return B2

def recover_B3( d, N, y, X ):
    """Extract B3 by projecting onto various q"""
    l = 1.0

    indices = []
    for i in xrange(d):
        for j in xrange(i, d):
            for k in xrange(j, d):
                indices.append( (i, j, k) )
    d_ = len(indices)
    indices = zip(* indices)

    # Pick a random set of d_ x's from the X
    X0 = X[:d_]
    Q = exp( - cdist( X, X0 )**2 / l )

    # The transform
    Theta = zeros( (d_, d_ ) )
    for i in xrange( d_ ):
        q = Q.T[i]
        Theta[i] = TriplesQ(X, q)[indices]
    B3_ = (y**3).dot(Q)/N

    B3 = zeros( (d, d, d) )
    B3[  indices ] = inv(Theta).dot(B3_)
    # Ugly 
    for i in xrange(d):
        for j in xrange(i, d):
            for k in xrange(j, d):
                if i == j and j == k:
                    pass
                elif i != j and j != k:
                    B3[i,j,k] /= 3
                else:
                    B3[i,j,k] /= 2
    for i in xrange(d):
        for j in xrange(d):
            for k in xrange(d):
                idx = [i,j,k]
                idx.sort()
                i_, j_, k_ = idx
                B3[i,j,k] = B3[i_,j_,k_]  
    #B3 = ( sc.swapaxes( B3, 0, 1 ) + sc.swapaxes( B3, 0, 2 ) + sc.swapaxes( B3, 1, 2 ) )/6

    return B3

def recover_B( k, d, B2, B3 ):
    """X2: vector -> matrix, while X3: vector -> tensor"""
    # Get singular vectors
    U, _, _ = svdk( B2, k )

    def tensordot( T, v ):
        return sc.einsum( 'abj,j ->ab', T, v )

    while True:
        # Get a random basis set
        theta = orthogonal( k )
        B2i = inv( U.T.dot( B2 ).dot( U ) ) 
        B123_ = sc.einsum( 'ijk,ia,jb,kc->abc', B3, U, U, U )
        B123 = sc.einsum( 'ajc,jb ->abc', B123_, B2i )

        l, R1 = eig( tensordot( B123, theta.T[0] ) )
        R1 = array( map( lambda col: col/norm(col), R1.T ) ).T
        assert( norm(R1.T[0]) - 1.0 < 1e-3 )

        # Restart
        if not ( sc.isreal( l ).all() ):
            continue

        L = [l.real]
        for i in xrange( 1, k ):
            l = diag( inv(R1).dot( tensordot( B123, theta.T[i] ).dot( R1 ) ) )
            # Restart
            if not ( sc.isreal( l ).all() ):
                continue
            L.append( l )
        L = array( sc.vstack( L ) )

        M3_ = U.dot( inv(theta.T) ).dot( L )
        return M3_

def normalise( y, X ):

    # Normalise data 
    # Center
    mu = X.mean(0)
    X = X - mu
    # Whiten
    S = Pairs( X, X )
    W = cholesky( S )
    X = X.dot( inv( W ) )

    # Normalise y

    return y, X, mu, S

def test_sample_recovery():
    """Test the accuracy of sample recovery"""
    K = 3
    d = 3
    N = 1e5

    fname = "/tmp/test.npz"
    lrm = LinearRegressionsMixture.generate(fname, K, d, cov = "eye", betas="eye")
    M, S, pi, B = lrm.mean, lrm.sigma, lrm.weights, lrm.betas

    y, X = lrm.sample( N )

    B2 = B.dot( diag( pi ) ).dot( B.T )
    B3 = sum( [ pi[i] * tensorify(B.T[i], B.T[i], B.T[i] ) for i in xrange(K) ] )

    B2_ = recover_B2( K, d, N, y, X )

    B3_ = recover_B3( d, N, y, X )

    B_ = recover_B( K, d, B2_, B3_ )
    B_ = closest_permuted_matrix( B.T, B_.T ).T

    err = norm( B - B_ )
    print "B:", err
    print B, B_

    assert( err < 1e-2 )

def test_discrete():
    """Test the accuracy of sample recovery"""
    K = 2
    d = 2

    # Simple orthogonal lines
    B = eye(d)[:,:K]
    #B = sc.randn(d,k)
    pi = ones(d)/d
    B2 = B.T.dot(diag(pi)).dot(B)
    B3 = sum( [ pi[i] * tensorify(B.T[i], B.T[i], B.T[i] ) for i in xrange(K) ] )

    N = 5
    X = sc.randn(N,d)

    # Recovering B2

    # The X2's are really the X's unrolled
    indices = sc.triu_indices(d)
    X2_0 = sc.column_stack( [ outer( x, x )[indices] for x in X ] )

    # For (d * d+1)/2 = 3 q's run the algorithm
    d_ = d * (d+1) / 2
    Q = dirichlet( ones(N), d_ )

    X2 = X2_0.dot( Q.T )
    assert( abs( det(X2) ) > 1e-6 )

    Y2 = X2.dot( B2[indices] )

    B2_ = zeros((d,d))
    B2_[indices] = inv(X2).dot( Y2 )
    B2_ = (B2_ + B2_.T) - diag(diag( B2_ ))
    print B2, B2_
    assert( sc.allclose( B2, B2_ ) )

    # Recovering B3
    indices = []
    for i in xrange(d):
        for j in xrange(i, d):
            for k in xrange(j, d):
                indices.append( (i, j, k) )

    d_ = len(indices) 
    indices = zip(* indices)

    X3_0 = sc.column_stack( [tensorify(x,x,x)[indices] for x in X ] )
    Q = dirichlet( ones(N), d_ )

    X3 = X3_0.dot( Q.T )
    assert( abs( det(X3) ) > 1e-6 )

    Y3 = X3.dot( B3[indices] )

    B3_ = zeros((d,d,d))
    B3_[indices] = inv(X3).dot(Y3)
    # appropriately divide to account for the symmetries
    B3_ = ( sc.swapaxes( B3_, 0, 1 ) + sc.swapaxes( B3_, 0, 2 ) + sc.swapaxes( B3_, 1, 2 ) ) 
    for i in xrange(d):
        B3_[i,i,i] /= 3
    print B3, B3_
    assert( sc.allclose( B3, B3_ ) )

    # Recovering B
    B_ = recover_B( K, d, B2_, B3_ )
    B_ = closest_permuted_matrix( B.T, B_.T ).T

    print B
    print B_

    assert( sc.allclose( B, B_ ) )

if __name__ == "__main__":
    import tempfile 

    K = 2
    d = 3
    N = 1e6

    sc.random.seed(0)

    # Initialise a model
    fname = tempfile.mktemp()
    lrm = LinearRegressionsMixture.generate(fname, K, d, cov = "eye", betas="eye")
    M, S, pi, B = lrm.mean, lrm.sigma, lrm.weights, lrm.betas

    # Generate some samples
    y, X = lrm.sample( N )

    # Add some noise to y
    withNoise = False
    if( withNoise ):
        sigma2 = 0.2
        noise = sc.randn(*y.shape) * sqrt( sigma2 )
        y += noise

    B2 = B.dot( diag( pi ) ).dot( B.T )
    B3 = sum( [ pi[i] * tensorify(B.T[i], B.T[i], B.T[i] ) for i in xrange(K) ] )

    B2_ = recover_B2( K, d, N, y, X )
    print "B2", norm(B2 - B2_)

    B3_ = recover_B3( d, N, y, X )
    print "B3:", norm(B3 - B3_)

    B_ = recover_B( K, d, B2_, B3_ )
    B_ = closest_permuted_matrix( B.T, B_.T ).T
    print "B:", norm( B - B_ )

