"""
Tests for the spectral package
"""

import scipy as sc 
from scipy import matrix, array, all, allclose, diag
from scipy.linalg import norm

import spectral.random as sr
import spectral.linalg as sl
import spectral.data as sd
import spectral.mixture 

from generators import gmm

def test_random_orthogonal( ):
    """Generate several matrices from orthogonal(), and verify that
    they are indeed orthogonal"""
    def check( d ):
        q = sr.orthogonal( d )
        assert allclose( q.dot( q.T ), sc.eye( d ) )

    for d in xrange( 2, 10 ):
        for i in xrange( 10 ):
            yield check, d

def test_permutation_inversion( ):
    """Generate several permutation from permutation(), and verify
    the inversion indeed returns the same list"""

    def check( x, perm ):
        n = len( x )
        perm_ = sl.invert_permutation( perm )

        y = sl.apply_permutation( perm_, sl.apply_permutation( perm, x ) )
        assert all( x == y )

        z = sl.apply_permutation( perm, sl.apply_permutation( perm_, x ) )
        assert all( x == z )

    for i in xrange( 10 ):
        n = sc.random.randint( 20 )
        x = sc.randn( n )
        perm = sr.permutation( n )

        yield check, x, perm

def test_matrix_permutation_shape( ):
    """Generate several matrix permutation from matrix_permutation(), and verify
    the inversion indeed returns the same list"""

    def check( x, perm ):
        m, n = x.shape
        y = sl.apply_matrix_permutation( perm, x )
        assert x.shape == y.shape

    for i in xrange( 10 ):
        m, n = sc.random.randint( 1, 20 ), sc.random.randint( 1, 20 )
        x = sc.randn( m, n )
        perm = sr.matrix_permutation( m, n )

        yield check, x, perm

def test_matrix_permutation_inversion( ):
    """Generate several matrix permutation from matrix_permutation(), and verify
    the inversion indeed returns the same list"""

    def check( x, perm ):
        m, n = x.shape
        perm_ = sl.invert_matrix_permutation( perm )
        x = sc.randn( m, n )
        y = sl.apply_matrix_permutation( perm_, sl.apply_matrix_permutation( perm, x ) )
        assert all( x == y )

        z = sl.apply_matrix_permutation( perm, sl.apply_matrix_permutation( perm_, x ) )
        assert all( x == z )

    for i in xrange( 10 ):
        m, n = sc.random.randint( 1, 20 ), sc.random.randint( 1, 20 )
        x = sc.randn( m, n )
        perm = sr.matrix_permutation( m, n )

        yield check, x, perm

def test_canonical_ordered( ):
    """Check if the property is idempotent"""

    def check( x ):
        y = sl.canonicalise( x )
        yr = y[0,:]
        yc = y[:,0]

        assert all( yr == sc.sort( yr ) )
        assert all( yc == sc.sort( yc ) )

    for i in xrange( 10 ):
        m, n = sc.random.randint( 1, 20 ), sc.random.randint( 1, 20 )
        x = sc.randn( m, n )

        yield check, x

def test_canonical_idempotence( ):
    """Check if the property is idempotent"""

    def check( x ):
        y = sl.canonicalise( x )
        z = sl.canonicalise( y )
        assert all( y == z )

    for i in xrange( 10 ):
        m, n = sc.random.randint( 1, 20 ), sc.random.randint( 1, 20 )
        x = sc.randn( m, n )

        yield check, x

def test_sample_moments( ):
    """Check that the moments of the data are close to the analytic values"""
    def check( k, d ):
        model = gmm.GaussianMixtureModel.generate( k, d )
        M1, M2, M3 = model.means
        w = model.weights

        x1, x2, x3 = model.sample( 1e5 )

        # Get the first moments of the data
        X1 = M1.dot( w )
        X2 = M2.dot( w )
        X3 = M3.dot( w )

        X1_ = x1.mean( axis=0 )
        X2_ = x2.mean( axis=0 )
        X3_ = x3.mean( axis=0 )

        err1 = norm( X1 - X1_) 
        err2 = norm( X2 - X2_) 
        err3 = norm( X3 - X3_) 
        print err1, err2, err3
        assert err1 < 1e-02
        assert err2 < 1e-02
        assert err3 < 1e-02

        # Get pairwise estimates
        P12, P13, P123 = spectral.mixture.exact_moments( w, M1, M2, M3 )

        P12_ = sd.Pairs( x1, x2 )
        P13_ = sd.Pairs( x1, x3 )

        err12 = norm( P12 - P12_)
        err13 = norm( P13 - P13_)
        print err12, err13
        assert err12 < 1e-02
        assert err13 < 1e-02

        eta = sc.randn( d )

        # Get triple estimates
        P123 = M1.dot(  diag( M3.T.dot(eta) * w ).dot( M2.T ) )
        P123_ = sd.Triples( x1, x2, x3, eta )

        err123 = norm( P123 - P123_) 
        print err123
        assert norm( P123 - P123_) < 1e-01

    for k in xrange( 2, 10 ):
        for d in xrange( k+1, 20, 3 ):
            for i in xrange( 3 ):
                yield check, k, d

def test_exact_recovery( ):
    def check( k, d ):
        model = gmm.GaussianMixtureModel.generate( k, d )
        M1, M2, M3 = model.means
        weights = model.weights

        M3_ = spectral.mixture.exact_recovery( weights, M1, M2, M3 )
        err = sd.recovery_error( M3, M3_ )
        assert err < 1e-05

    for k in xrange( 2, 10 ):
        for d in xrange( k+1, 20, 3 ):
            for i in xrange( 3 ):
                yield check, k, d

def test_fuzzed_exact_recovery( ):
    def check( k, d, fuzz ):
        model = gmm.GaussianMixtureModel.generate( k, d )
        M1, M2, M3 = model.means
        w = model.weights
        P12, P13, P123 = spectral.mixture.exact_moments( w, M1, M2, M3 )

        P12_ = P12 + sc.randn( d, d ) * fuzz
        P13_ = P13 + sc.randn( d, d ) * fuzz
        P123_ = lambda eta: P123(eta) + sc.randn( d, d )/500.0

        M3_ = spectral.mixture.recoverM3( k, P12_, P13_, P123_ )
        err = sd.recovery_error( M3, M3_ )
        print sl.canonicalise(M3), sl.canonicalise(M3_)
        print err
        assert err < 1e-05

    for k in xrange( 2, 10 ):
        for d in xrange( k+1, 10, 4 ):
            for fuzz in [ 1e-5, 1e-3, 1e-1 ]:
                yield check, k, d, fuzz
                
def test_sample_recovery( ):
    def check( k, d ):
        model = gmm.GaussianMixtureModel.generate( k, d )
        _, _, M3 = model.means

        x1, x2, x3 = model.sample( 1e5 )

        M3_ = spectral.mixture.sample_recovery( k, x1, x2, x3 )
        err = sd.recovery_error( M3, M3_ )
        print sl.canonicalise(M3), sl.canonicalise(M3_)
        print err
        assert err < 1e-02

    for k in [2, 3, 5, 10]:
        for d in xrange( k+1, 20, 4 ):
            for i in xrange( 1 ):
                yield check, k, d

if __name__ == "__main__":
    print "Testing exact recovery..."
    for tst in test_exact_recovery():
        tst[0]( *tst[1:] )
    print "Testing fuzzed exact recovery..."
    for tst in test_fuzzed_exact_recovery():
        tst[0]( *tst[1:] )
    print "Testing sample recovery..."
    for tst in test_sample_recovery():
        tst[0]( *tst[1:] )

