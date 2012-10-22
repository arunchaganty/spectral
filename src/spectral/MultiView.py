"""
Spectral decomposition method from
Anandkumar, Hsu, Kakade, "A Method of Moments for Mixture Models and
Hidden Markov Models" (2012).
"""

import ipdb
import scipy as sc 
from scipy import diag, array, ndim, outer, eye, ones, log
from scipy.linalg import norm, svd, svdvals, eig, eigvals, inv, det
from spectral.linalg import svdk, mrank, approxk, eigen_sep, \
        closest_permuted_matrix, tensorify, matrix_tensorify, \
        column_aerr, column_rerr
from spectral.rand import orthogonal
from spectral.data import Pairs, Triples
from spectral.util import DataLogger
from generators import MultiViewGaussianMixtureModel

import time

logger = DataLogger("log")

def recover_M3( k, P12, P13, P123, P12e, P13e, P123e, delta=0.01 ):
    """Recover M3 from P_12, P_13 and P_123"""
    d, _ = P12.shape

    # Inputs
    logger.add_err( "P12", P12e, P12, 2 )
    logger.add_consts( "P12", P12e, k, 2 )
    logger.add_err( "P13", P12e, P12, 2 )
    logger.add_consts( "P13", P12e, k, 2 )
    logger.add_err( "P123", P123e, P123 )
    logger.add_consts( "P123", P123e )

    # Get singular vectors
    U1, _, U2 = svdk( P12, k )
    _, _, U3 = svdk( P13, k )
    U2, U3 = U2.T, U3.T

    U1e, _, U2e = svdk( P12, k )
    _, _, U3e = svdk( P13, k )
    U2e, U3e = U2e.T, U3e.T

    # Check U_1.T P_{12} U_2 is invertible
    assert( sc.absolute( det( U1.T.dot( P12 ).dot( U2 ) ) ) > 1e-16 )

    while True:
        # Get a random basis set
        theta = orthogonal( k )

        P12i = inv( U1.T.dot( P12 ).dot( U2 ) ) 
        B123_ = sc.einsum( 'ijk,ia,jb,kc->abc', P123, U1, U2, U3 )
        B123 = sc.einsum( 'ajc,jb ->abc', B123_, P12i )

        P12ie = inv( U1e.T.dot( P12e ).dot( U2e ) ) 
        B123e_ = sc.einsum( 'ijk,ia,jb,kc->abc', P123e, U1e, U2e, U3e )
        B123e = sc.einsum( 'ajc,jb ->abc', B123e_, P12ie )

        logger.add_err( "B123", B123, B123e )

        l, R1 = eig( B123.dot( theta.T[0] ) )
        R1 = array( map( lambda col: col/norm(col), R1.T ) ).T
        assert( norm(R1.T[0]) - 1.0 < 1e-10 )

        le, R1e = eig( B123e.dot( theta.T[0] ) )
        logger.add_err( "R", R1e, R1, 2 )
        logger.add_consts( "R", R1e, k, 2 )

        # Restart
        if not ( sc.isreal( l ).all() ):
            continue

        L = [l.real]
        for i in xrange( 1, k ):
            l = diag( inv(R1).dot( B123.dot( theta.T[i] ).dot( R1 ) ) )
            # Restart
            if not ( sc.isreal( l ).all() ):
                continue
            L.append( l )
        L = array( sc.vstack( L ) )

        Le = [le.real]
        for i in xrange( 1, k ):
            le = diag( inv(R1e).dot( B123e.dot( theta.T[i] ).dot( R1e ) ) )
            Le.append( le )
        Le = array( sc.vstack( Le ) )
        logger.add_err( "L", Le, L, 2 )

        M3_ = U3.dot( inv(theta.T) ).dot( L )
        logger.add( "M3_", M3_ )
        return M3_

def exact_moments( w, M1, M2, M3 ):
    """Tests the algorithm with exact means"""
    assert( ndim( w ) == 1 )

    # Input processing
    k = w.shape[0]
    M1, M2, M3 = M1, M2, M3 
    assert( mrank(M1) == k )
    assert( mrank(M2) == k )
    assert( mrank(M3) == k )

    # Get pairwise estimates
    P12 = M1.dot( diag( w ).dot( M2.T ) )
    P13 = M1.dot( diag( w ).dot( M3.T ) )
    P123 = sum( [ w[i] * tensorify( M1.T[i], M2.T[i], M3.T[i] ) for i in xrange( k ) ] )

    return P12, P13, P123

def test_exact_recovery():
    """Test the accuracy of exact recovery"""
    mvgmm = sc.load( "test-data/mvgmm-2-3-1e4.npz" )
    k, M, w = mvgmm['k'], mvgmm['M'], mvgmm['w']

    M1, M2, M3 = M

    P12, P13, P123 = exact_moments( w, M1, M2, M3 )

    M3_ = recover_M3( k, P12, P13, P123, P12, P13, P123 )
    M3_ = closest_permuted_matrix( M3.T, M3_.T ).T

    print norm( M3 - M3_ )/norm( M3 )
    print M3
    print M3_

    assert norm(M3 - M3_)/norm( M3 ) < 1e-2

def sample_moments( x1, x2, x3 ):
    """Learn a model using SVD and three views with k vectors"""

    assert( x1.shape == x2.shape and x2.shape == x3.shape )
    return Pairs( x1, x2 ), Pairs( x1, x3 ), Triples( x1, x2, x3 )

def test_sample_recovery():
    """Test the accuracy of recovery with actual samples"""
    mvgmm = sc.load( "test-data/mvgmm-2-3-1e4.npz" )
    k, M, w, X = mvgmm['k'], mvgmm['M'], mvgmm['w'], mvgmm['X']

    M1, M2, M3 = M

    X1, X2, X3 = X

    P12, P13, P123 = sample_moments( X1, X2, X3 )
    P12e, P13e, P123e = exact_moments( w, M1, M2, M3 )

    M3_ = recover_M3( k, P12, P13, P123, P12e, P13e, P123e )
    M3_ = closest_permuted_matrix( M3.T, M3_.T ).T

    print norm( M3 - M3_ )/norm( M3 )
    print M3
    print M3_

    assert norm(M3 - M3_)/norm( M3 ) < 1e-2

def main( fname, samples, delta ):
    """Run on sample in fname"""

    mvgmm = sc.load( fname )
    k, d, M, w, X = mvgmm['k'], mvgmm['d'], mvgmm['M'], mvgmm['w'], mvgmm['X']

    (M1, M2, M3) = M
    logger.add( "M3", M3 )
    logger.add_consts( "M1", M1, k, 2 )
    logger.add_consts( "M2", M2, k, 2 )
    logger.add_consts( "M3", M3, k, 2 )
    logger.add_consts( "w_min", w.min() )
    logger.add_consts( "w_max", w.max() )

    (X1, X2, X3) = X

    N, _ = X1.shape
    if (samples < 0 or samples > N):
        print "Warning: %s greater than number of samples in file. Using %s instead." % ( samples, N )
    else:
        X1, X2, X3 = X1[:samples, :], X2[:samples, :], X3[:samples, :]
        X = X1, X2, X3

    P12, P13, P123 = sample_moments( X1, X2, X3 )
    P12e, P13e, P123e = exact_moments( w, M1, M2, M3 )

    start = time.time()
    M3_ = recover_M3( k, P12, P13, P123, P12e, P13e, P123e, delta=delta )
    stop = time.time()
    M3_ = closest_permuted_matrix( M3.T, M3_.T ).T

    # Error data
    logger.add_err( "M3", M3, M3_ )
    logger.add_err( "M3", M3, M3_, 'col' )
    logger.add( "time", stop - start )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "fname", help="Input file (as npz)" )
    parser.add_argument( "ofname", help="Output file (as npz)" )
    parser.add_argument( "--seed", default=time.time(), type=long, help="Seed used" )
    parser.add_argument( "--samples", default=-1, type=float, help="Number of samples to be used" )
    parser.add_argument( "--delta", default=0.01, type=float, help="Confidence bound" )

    args = parser.parse_args()

    logger = DataLogger(args.ofname)

    print "Seed:", int( args.seed )
    sc.random.seed( int( args.seed ) )

    logger.add( "seed", int( args.seed ) )

    main( args.fname, int(args.samples), args.delta )

