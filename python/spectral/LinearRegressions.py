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

def recover_B2( k, d, N, X2, Y2 ):
    """X2: vector -> matrix, while X3: vector -> tensor"""
    # Extract B2 by projecting on a large number of S 
    d_ = d * (d+1) / 2
    B2 = zeros( d_ )

    # The transform
    T2 = zeros( (d_, d_) )

    # Generate the qs
    Q = dirichlet( 0.5 * ones(N), d_ )

    B2_ = zeros( d_ )

    indices = sc.triu_indices( d )

    for i in xrange( d_ ):
        q = Q[i]
        T2[i] = X2(q)[indices]
        B2_[i] = Y2(q)

    B2 = zeros( (d, d) )
    B2[ indices ] = inv(T2).dot(B2_)
    B2 = (B2 + B2.T)/2

    return B2

def recover_B3( k, d, N, X3, Y3 ):
    """X2: vector -> matrix, while X3: vector -> tensor"""

    indices = []
    for i in xrange(d):
        indices.append( (i, i, i ) )
        for j in xrange(i+1, d):
            indices.append( (i, i, j ) )
            for k in xrange(j+1, d):
                indices.append( (i, j, k) )
    indices = zip(* indices)
    # Extract B2 by projecting on a large number of S 
    d_ = d * (d**2 + 5)/6
    B3 = zeros( d_ )

    # The transform
    T3 = zeros( (d_, d_ ) )

    # Generate the qs
    Q = dirichlet( 0.5 * ones(N), d_ )


    B3_ = zeros( d_ )
    for i in xrange( d_ ):
        q = Q[i]
        T3[i] = X3(q)[indices]
        B3_[i] = Y3(q)

    B3 = zeros( (d, d, d) )
    B3[  indices ] = inv(T3).dot(B3_)
    B3 = ( sc.swapaxes( B3, 0, 1 ) + sc.swapaxes( B3, 0, 2 ) + sc.swapaxes( B3, 1, 2 ) )/6

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
    N = 1e6

    fname = "/tmp/test.npz"
    lrm = LinearRegressionsMixture.generate(fname, k, d, cov = "eye")
    M, S, pi, B = lrm.mean, lrm.sigma, lrm.weights, lrm.betas

    y, X = lrm.sample( N )

    B2 = B.dot( diag( pi ) ).dot( B.T )

    X2, X3, Y2, Y3 = sample_moments( k, y, X )

    B2_ = recover_B2( k, d, N, X2, Y2 )
    B3_ = recover_B3( k, d, N, X3, Y3 )
    B_ = recover_B( k, d, B2_, B3_ )
    #B2_ = closest_permuted_matrix( B2.T, B2_.T ).T
    #B2_ = closest_permuted_matrix( B2.T, B2_.T ).T
    print B2
    print B2_

    B_ = closest_permuted_matrix( B.T, B_.T ).T

    print ( norm( B - B_ ) )
    print B
    print B_

    assert( sc.allclose( B2, B2_ ) )

def test_discrete():
    """Test the accuracy of sample recovery"""
    k = 2
    d = 3

    # Simple orthogonal lines
    B = eye(d)[:,:k]
    pi = ones(d)/d
    B2 = B.T.dot(diag(pi)).dot(B)
    B3 = sum( [ pi[i] * tensorify(B.T[i], B.T[i], B.T[i] ) for i in xrange(k) ] )

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
    assert( det(X2) > 1e-6 )

    Y2 = X2.dot( B2[indices] )

    B2_ = zeros((d,d))
    B2_[indices] = inv(X2).dot( Y2 )
    B2_ = (B2_ + B2_.T) - diag(diag( B2_ ))
    assert( sc.allclose( B2[indices], B2_[indices] ) )

    # Recovering B3
    indices = []
    for i in xrange(d):
        indices.append( (i, i, i ) )
        for j in xrange(i+1, d):
            indices.append( (i, i, j ) )
            for k in xrange(j+1, d):
                indices.append( (i, j, k) )
    indices = zip(* indices)

    d_ = d * (d**2 + 5)/6
    B3 = zeros( d_ )
    X3_0 = sc.column_stack( [tensorify(x,x,x)[indices] for x in X ] )
    Q = dirichlet( ones(N), d_ )

    X3 = X3_0.dot( Q.T )
    assert( det(X3) > 1e-6 )

    Y3 = X3.dot( B3[indices] )

    B3_ = zeros((d,d,d))
    B3_[indices] = inv(X3).dot(Y3)
    B3_ = ( sc.swapaxes( B3_, 0, 1 ) + sc.swapaxes( B3_, 0, 2 ) + sc.swapaxes( B3_, 1, 2 ) )/6
    assert( sc.allclose( B3[indices], B3_[indices] ) )


