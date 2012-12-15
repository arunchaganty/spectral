"""
Quick implementation of the mixture of linear regressions code
"""

import shutil, tempfile 

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
from spectral.rand import orthogonal, wishart, dirichlet, multinomial
from spectral.data import Pairs, Triples, PairsQ, TriplesQ
from models import LinearRegressionsMixture 

def getQ( n, X, mode, *args ):
    N, d = X.shape

    # Pick our set of Q
    if mode == "local":
        # Pick a random set of d_ x's from the X
        # And locally reweight points based on their distance 
        if len(args) == 0: 
            l = 1.0
        else:
            l = args[0]
            args = args[1:]
        X0 = X[:n]
        Q = exp( - cdist( X0, X )**2 / l )
    elif mode == "dirichlet":
        # Pick a random set of points and assign weights
        if len(args) == 0: 
            alpha = 1.0
        else:
            alpha = args[0]
            args = args[1:]
        Q = dirichlet( alpha * ones(N), n )
    elif mode == "subset":
        # Pick a random set of points and assign weights
        if len(args) == 0: 
            subset_size = sc.ceil( 0.01 * N ) # 1%
        else:
            subset_size = sc.ceil( args[0] * N )
            args = args[1:]
        Q = []
        for i in xrange( n ):
            q = array( multinomial( subset_size, ones( N )/N), dtype = sc.double ) 
            Q.append( q )
        Q = array( Q )
    else:
        raise NotImplementedError()
    # Normalize Q
    Q = (Q.T / Q.sum(1)).T

    assert (Q.shape == (n, N))
    return Q


def recover_B2( d, N, oversample, y, X, mode = "dirichlet", *args ):
    """ Extract B2 by projecting onto various q """

    indices = sc.triu_indices( d )
    # Handle noise by including another row
    d_ = (d * (d+1) / 2) 

    n_points = int( oversample * d_ )

    Q = getQ( n_points, X, mode, *args )

    # The transform
    Theta = zeros( (n_points, d_) )
    for i in xrange( n_points ):
        q = Q[i]
        Theta[i,:d_] = PairsQ(X, q)[indices]
    B2_ = (y**2).dot(Q.T)

    if len( args ) > 0:
        reg = args[0]
    else:
        reg = 0.0
    theta_inv = inv( Theta.T.dot( Theta ) + reg * eye( d_ ) ).dot( Theta.T )

    B2 = zeros( (d, d) )
    B2[ indices ] = theta_inv.dot(B2_)
    B2 = (B2 + B2.T)/2

    return B2

def recover_B3( d, N, oversample, y, X, mode = "dirichlet", *args ):
    """Extract B3 by projecting onto various q"""
    l = 1.0

    indices = []
    for i in xrange(d):
        for j in xrange(i, d):
            for k in xrange(j, d):
                indices.append( (i, j, k) )
    d_ = len(indices) 
    indices = zip(* indices)

    n_points = int( oversample * d_ )

    Q = getQ( n_points, X, mode, *args )

    # The transform
    Theta = zeros( (n_points, d_ ) )
    for i in xrange( n_points ):
        q = Q[i]
        Theta[i, :d_] = TriplesQ(X, q)[indices]
    B3_ = (y**3).dot(Q.T)

    if len( args ) > 0:
        reg = args[0]
    else:
        reg = 0.0
    theta_inv = inv( Theta.T.dot( Theta ) + reg * eye( d_ ) ).dot( Theta.T )

    B3 = zeros( (d, d, d) )
    B3[  indices ] = theta_inv.dot(B3_)
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

    B2_ = recover_B2( d, N, 1.0, y, X )
    B3_ = recover_B3( d, N, 1.0, y, X )
    B_ = recover_B( K, d, B2_, B3_ )
    B_ = closest_permuted_matrix( B.T, B_.T ).T

    err = norm( B - B_ )
    print "B:", err
    print B, B_

    assert( err < 1e-2 )

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

    # Add some noise to y
    if args.with_noise:
        sigma2 = 0.2
        noise = sc.randn(*y.shape) * sqrt( sigma2 )
        y += noise

    B2_ = recover_B2( d, N, args.oversample, y, X, args.mode, *args.args )
    B3_ = recover_B3( d, N, args.oversample, y, X, args.mode, *args.args )
    B_ = recover_B( K, d, B2_, B3_ )
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
    parser.add_argument( "--samples", default=1e6, type=float, help="Number of samples to be used" )
    parser.add_argument( "--with-noise", default=False, type=bool, help="Use noise" )
    parser.add_argument( "--mode", default="dirichlet", type=str, help="Generation of Q = dirichlet|local" )
    parser.add_argument( "--oversample", default=1.0, type=float, help="Use more points to regularise" )
    parser.add_argument( "args", metavar='args', type=float, nargs='+',
                       help='Arguments for the Q generation')

    args = parser.parse_args()
    main( args )

