"""
Spectral learning for mixture models
"""

import ipdb
import scipy as sc 
from scipy import diag, array, ndim
from scipy.linalg import norm, det, eig, svd, svdvals, inv
from spectral.linalg import svdk, mrank
from spectral.rand import orthogonal
from spectral.data import Pairs, Triples

def recoverM3( k, P12, P13, P123 ):
    """Recover M3 from P_12, P_13 and P_123"""
    # Get singular vectors
    U1, _, U2 = svdk( P12, k )
    _, _, U3 = svdk( P13, k )
    U2, U3 = U2.T, U3.T

    # Check U_1.T P_{12} U_2 is invertible
    assert( sc.absolute( det( U1.T.dot( P12 ).dot( U2 ) ) ) > 1e-16 )

    while True:
        # Get a random basis set
        theta = orthogonal( k )
        eta = U3.dot( theta ).T

        # Get the eigen value matrix L
        B123 = lambda eta_: U1.T.dot( P123( eta_ ) ).dot( U2 ).dot( inv( U1.T.dot( P12 ). dot( U2 ) ) )

        l, R1 = eig( B123( eta[0] ) )
        R1 = array( map( lambda col: col/norm(col), R1.T ) ).T
        assert( norm(R1.T[0]) - 1.0 < 1e-10 )

        # Restart
        if not ( sc.isreal( l ).all() ): continue

        L = [l.real]
        for i in xrange( 1, k ):
            l = diag( inv(R1).dot( B123( eta[i] ).dot( R1 ) ) )
            # Restart
            if not ( sc.isreal( l ).all() ): continue
            L.append( l )
        L = array( sc.vstack( L ) )

        M3_ = U3.dot( inv(theta.T) ).dot( L )
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
    P123 = lambda eta: \
        M1.dot( diag( M3.T.dot(eta) * w ) ).dot( M2.T ) 

    return P12, P13, P123

def exact_recovery( w, M1, M2, M3 ):
    """Tests the algorithm with exact means"""
    k = w.shape[0]
    P12, P13, P123 = exact_moments( w, M1, M2, M3 )
    return recoverM3( k, P12, P13, P123 )

def sample_moments( x1, x2, x3 ):
    """Learn a model using SVD and three views with k vectors"""

    assert( x1.shape == x2.shape and x2.shape == x3.shape )
    return Pairs( x1, x2 ), Pairs( x1, x3 ), Triples( x1, x2, x3 )

def sample_recovery( k, x1, x2, x3 ):
    """Learn a model using SVD and three views with k vectors"""
    P12, P13, P123 = sample_moments( x1, x2, x3 )

    assert( x1.shape == x2.shape and x2.shape == x3.shape )
    return recoverM3( k, P12, P13, P123 )

