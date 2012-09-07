"""
Tests for the spectral package
"""

import scipy as sc 
from scipy import diag, matrix, array, rank
from scipy.linalg import norm, det, eig, svd

import spectral.random as sr
import spectral.linalg as sl

import unittest 

class NumPyTestCase( unittest.TestCase ):
    """A NumPy test case, with several functions me suitable to numpy-written programs"""

    def assertAllClose( self, x, y, rtol=1e-05, atol=1e-08, msg=None ):
        """Assert all the variables are close, i.e. within epsilon"""
        if not sc.allclose( x, y ):
            standardMsg = "%s != %s" % ( repr( x ), repr( y ) ) 
            raise self.failureException( msg or standardMsg )

    def assertAll( self, x, y, msg=None ):
        """Assert all the variables are close, i.e. within epsilon"""
        if not sc.all( x == y ):
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
                self.assertAllClose( q.dot( q.T ), sc.eye( d ) )

    def test_permutation_inversion( self ):
        """Generate several permutation from permutation(), and verify
        the inversion indeed returns the same list"""

        for i in xrange( 10 ):
            n = sc.random.randint( 20 )
            perm = sr.permutation( n )
            perm_ = sl.invert_permutation( perm )

            x = sc.randn( n )
            y = sl.apply_permutation( perm_, sl.apply_permutation( perm, x ) )
            self.assertAll( x, y )

            z = sl.apply_permutation( perm, sl.apply_permutation( perm_, x ) )
            self.assertAll( x, z )

    def test_matrix_permutation_shape( self ):
        """Generate several matrix permutation from matrix_permutation(), and verify
        the inversion indeed returns the same list"""

        for i in xrange( 10 ):
            m, n = sc.random.randint( 1, 20 ), sc.random.randint( 1, 20 )
            perm = sr.matrix_permutation( m, n )

            x = sc.randn( m, n )
            y = sl.apply_matrix_permutation( perm, x )
            self.assertEqual( x.shape, y.shape )

    def test_matrix_permutation_inversion( self ):
        """Generate several matrix permutation from matrix_permutation(), and verify
        the inversion indeed returns the same list"""

        for i in xrange( 10 ):
            m, n = sc.random.randint( 1, 20 ), sc.random.randint( 1, 20 )
            perm = sr.matrix_permutation( m, n )
            perm_ = sl.invert_matrix_permutation( perm )

            x = sc.randn( m, n )
            y = sl.apply_matrix_permutation( perm_, sl.apply_matrix_permutation( perm, x ) )
            self.assertAll( x, y )

            z = sl.apply_matrix_permutation( perm, sl.apply_matrix_permutation( perm_, x ) )
            self.assertAll( x, z )

    def test_canonical_ordered( self ):
        """Check if the property is idempotent"""

        for i in xrange( 10 ):
            m, n = sc.random.randint( 1, 20 ), sc.random.randint( 1, 20 )
            x = sc.randn( m, n )
            y = sl.canonicalise( x )
            yr = y[0,:]
            yc = y[:,0]

            self.assertAll( yr, sc.sort( yr ) )
            self.assertAll( yc, sc.sort( yc ) )

    def test_canonical_idempotence( self ):
        """Check if the property is idempotent"""

        for i in xrange( 10 ):
            m, n = sc.random.randint( 1, 20 ), sc.random.randint( 1, 20 )
            x = sc.randn( m, n )
            y = sl.canonicalise( x )
            z = sl.canonicalise( y )
            self.assertAll( y, z )

