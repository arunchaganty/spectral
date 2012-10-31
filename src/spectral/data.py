"""
Data manipulation functions
"""

import scipy as sc
from scipy import matrix, array
from scipy.linalg import norm 

from . import _data

def count_frequency( X, d ):
    N, W = X.shape
    Y = _data.count_frequency( X, d )
    return Y/float(W)

Pairs = _data.Pairs
Triples = _data.Triples

def TriplesP(X1, X2, X3, theta):
    theta = sc.float32( theta ) 
    return _data.TriplesP( X1, X2, X3, theta )

def test_pairs():
    """Test Pairs"""
    
    # Generate N random vectors
    N = 1000
    d = 10

    X1 = sc.float32( sc.random.rand( N, d ) )
    X2 = sc.float32( sc.random.rand( N, d ) )

    P = Pairs( X1, X2 )

    P_ = sc.zeros( (d,d) )
    for i in xrange( N ):
        P_ += (sc.outer(X1[i], X2[i]) - P_)/(i+1)

    assert sc.allclose( P, P_ )

def test_triplesp():
    """Test Pairs"""
    
    # Generate N random vectors
    N = 1000
    d = 10

    X1 = sc.float32( sc.random.rand( N, d ) )
    X2 = sc.float32( sc.random.rand( N, d ) )
    X3 = sc.float32( sc.random.rand( N, d ) )

    theta = sc.random.rand( d )

    T = TriplesP( X1, X2, X3, theta )

    T_ = sc.zeros( (d,d) )
    for i in xrange( N ):
        T_ += (sc.outer(X1[i], X2[i]) * theta.dot(X3[i]) - T_)/(i+1)

    assert sc.allclose( T, T_ )

