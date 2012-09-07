"""
Tests for the spectral package
"""

import scipy as m 
from scipy import diag, matrix, array, rank
from scipy.linalg import norm, det, eig, svd

import spectral.random as sr
import spectral.linalg as sl

import unittest 

class NumPyTestCase( unittest.TestCase ):
    """A NumPy test case, with several functions me suitable to numpy-written programs"""

    def assertAllClose( self, x, y, rtol=1e-05, atol=1e-08, msg=None ):
        """Assert all the variables are close, i.e. within epsilon"""
        if not m.allclose( x, y ):
            standardMsg = "%s != %s" % ( repr( x ), repr( y ) ) 
            raise self.failureException( msg or standardMsg )

    def assertAll( self, x, y, msg=None ):
        """Assert all the variables are close, i.e. within epsilon"""
        if not m.all( x == y ):
            standardMsg = "%s != %s" % ( repr( x ), repr( y ) ) 
            raise self.failureException( msg or standardMsg )


class TestLinearAlgebra( NumPyTestCase ):
    """Test various linear algebra routines"""

    def test_random_orthogonal( self ):
        """Generate several matrices from orthogonal(), and verify that
        they are indeed orthogonal"""

        for d in xrange( 2, 10 ):
            for i in xrange( 10 ):
                q = sr.orthogonal( d )
                self.assertAllClose( q.dot( q.T ), m.eye( d ) )

    def test_permutation_inversion( self ):
        """Generate several permutation from permutation(), and verify
        the inversion indeed returns the same list"""

        for i in xrange( 10 ):
            n = m.random.randint( 20 )
            perm = sr.permutation( n )
            perm_ = sl.invert_permutation( perm )

            x = m.randn( n )
            y = sl.apply_permutation( perm_, sl.apply_permutation( perm, x ) )
            self.assertAll( x, y )

            z = sl.apply_permutation( perm, sl.apply_permutation( perm_, x ) )
            self.assertAll( x, z )


