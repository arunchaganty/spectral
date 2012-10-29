"""
E-M algorithm to detect and separate multiview 
"""

import em
import scipy as sc
import scipy.misc
import scipy.spatial
import scipy.linalg

from scipy import array, eye, ones, log
from scipy.linalg import norm
cdist = scipy.spatial.distance.cdist
multivariate_normal = scipy.random.multivariate_normal
logsumexp = scipy.logaddexp.reduce

from spectral.linalg import closest_permuted_matrix, \
        column_aerr, column_rerr
from util import DataLogger
from models import MultiViewGaussianMixtureModel

logger = DataLogger("log")

class MultiViewGaussianMixtureEM( em.EMAlgorithm ):
    """A multiview gaussian, each with spherical covariance"""
    def __init__( self, k, d ):
        self.k, self.d = k, d
        em.EMAlgorithm.__init__( self )

    def compute_expectation( self, X, O ):
        """Compute the most likely values of the latent variables; returns lhood"""
        X1, X2, X3 = X
        (N, d), k = X1.shape, self.k
        (M1, M2, M3), (S1, S2, S3), w = O

        total_lhood = 0
        # Get pairwise distances between centers (D_ij = \|X_i - M_j\|)
        D1 = cdist( X1, M1.T )
        D2 = cdist( X2, M2.T )
        D3 = cdist( X3, M3.T )

        # Log-likelihood = - 0.5 ( S1^-2 D1**2 + S2^-2 D2**2 + S2^-2
        # D2**2) + log w + -d/2 (log S1 + log S2 + log S3)
        Z = -0.5 * (D1**2/S1**2 + D2**2/S2**2 + D3**2/S3**2) + log( w ) - 0.5 * d * (log(S1) + log(S2) + log(S3))
        total_lhood += logsumexp( logsumexp(Z) )

        # Normalise the probilities (soft EM)
        Z = sc.exp(Z.T - logsumexp(Z, 1)).T
            
        return total_lhood, Z

    def compute_maximisation( self, X, Z, O ):
        """Compute the most likely values of the parameters"""

        X1, X2, X3 = X
        (N, d), k = X1.shape, self.k
        (M1, M2, M3), (S1, S2, S3), w = O

        # Cluster weights
        w = Z.sum(axis=0) + 1

        # Get new means
        M1 = (Z.T.dot( X1 ).T / w)
        M2 = (Z.T.dot( X2 ).T / w)
        M3 = (Z.T.dot( X3 ).T / w)

        S1 = (cdist( X1, M1.T ) * Z).sum()/(d*N)
        S2 = (cdist( X2, M2.T ) * Z).sum()/(d*N)
        S3 = (cdist( X3, M3.T ) * Z).sum()/(d*N)

        w /= w.sum()

        return (M1, M2, M3), (S1, S2, S3), w

    def kmeanspp_initialisation( self, X ):
        """Initialise means using K-Means++"""
        N, _ = X.shape
        k, d = self.k, self.d
        M = []

        # Choose one center amongst the X at random
        m = sc.random.randint( N )
        M.append( X[m] )

        # Choose k centers
        while( len( M ) < self.k ):
            # Create a probability distribution D^2 from the previous mean
            D = cdist( X, M ).min( 1 )**2
            assert( D.shape == (N,) )

            # Normalise and sample a new point
            D /= D.sum()

            m = sc.random.multinomial( 1, D ).argmax()
            M.append( X[m] )

        M = sc.column_stack( M )
        sigma = cdist( X, M.T ).sum()/(k*d*N)
        w = ones( k )/float(k)

        return M, sigma, w

    def run( self, X, O = None, *args, **kwargs ):
        if O == None:
            X1, X2, X3 = X
            M1, S1, w = self.kmeanspp_initialisation( X1 )
            M2, S2, _ = self.kmeanspp_initialisation( X2 )
            M3, S3, _ = self.kmeanspp_initialisation( X3 )
            O = (M1, M2, M3), (S1, S2, S3), w
        return em.EMAlgorithm.run( self, X, O, *args, **kwargs )

def test_multiview_gmm_em():
    fname = "./test-data/mvgmm-3-10-1e6"
    mvgmm = MultiViewGaussianMixtureModel.from_file( fname )
    k, d, M, S, v, w = mvgmm.k, mvgmm.d, mvgmm.means, mvgmm.sigmas, mvgmm.n_views, mvgmm.weights
    # Simplifying assumption for now
    X = []
    for i in xrange( v ):
        X.append( mvgmm.get_samples( "X%d" % (i+1), d ) )

    assert( v == 3 )

    algo = MultiViewGaussianMixtureEM( k, d )

    M1, M2, M3 = M
    O = M, S, w

    start = time.time()
    def report( i, O_, lhood ):
        (_, _, M3_), _, _ = O_
        logger.add_err( "M_3_t%d" % (i), M3, M3_ )
        logger.add( "time_%d" % (i), time.time() - start )
    lhood, Z, O_ = algo.run( X, None, report )
    logger.add( "time", time.time() - start )

    (M1_, M2_, M3_), (S1, S2, S3), w = O_

    M1_ = closest_permuted_matrix( M1, M1_ )
    M2_ = closest_permuted_matrix( M2, M2_ )
    M3_ = closest_permuted_matrix( M3, M3_ )

    print column_aerr( M3, M3_ ), column_rerr( M3, M3_ )

    assert column_rerr( M3, M3_ ) < 1e-2

def main(fname, samples):
    """Run MVGMM EM on the data in @fname"""

    mvgmm = MultiViewGaussianMixtureModel.from_file( fname )
    k, d, M, S, v, w = mvgmm.k, mvgmm.d, mvgmm.means, mvgmm.sigmas, mvgmm.n_views, mvgmm.weights

    # Simplifying assumption for now
    assert( v == 3 )
    X = []
    for i in xrange( v ):
        X.append( mvgmm.get_samples( "X%d" % (i+1), d ) )
    algo = MultiViewGaussianMixtureEM( k, d )

    X1, X2, X3 = X
    M1, M2, M3 = M
    N, _ = X1.shape

    if (samples < 0 or samples > N):
        print "Warning: %s greater than number of samples in file. Using\
        %s instead." % ( samples, N )
    else:
        X1, X2, X3 = X1[:samples, :], X2[:samples, :], X3[:samples, :] 
        X = (X1, X2, X3)
    N, _ = X1.shape

    O = M, S, w
    
    start = time.time()
    def report( i, O_, lhood ):
        (_, _, M3_), _, _ = O_
        logger.add_err( "M_3_t%d" % (i), M3, M3_ )
        logger.add( "time_%d" % (i), time.time() - start )
    lhood, Z, O_ = algo.run( X, None, report )
    logger.add( "time", time.time() - start )
    
    (M1_, M2_, M3_), (S1, S2, S3), w = O_

    logger.add( "k", k )
    logger.add( "d", d )
    logger.add( "N", N )
    logger.add( "M1", M1 )
    logger.add( "M2", M2 )
    logger.add( "M3", M3 )

    M1_ = closest_permuted_matrix( M1.T, M1_.T ).T
    M2_ = closest_permuted_matrix( M2.T, M2_.T ).T
    M3_ = closest_permuted_matrix( M3.T, M3_.T ).T
    logger.add( "M1_", M1_ )
    logger.add( "M2_", M2_ )
    logger.add( "M3_", M3_ )

    logger.add_err( "M", M1, M1_ )
    logger.add_err( "M", M1, M1_, 'col' )
    logger.add_err( "M", M2, M2_ )
    logger.add_err( "M", M2, M2_, 'col' )
    logger.add_err( "M", M3, M3_ )
    logger.add_err( "M", M3, M3_, 'col' )
    print column_aerr( M3, M3_ ), column_rerr( M3, M3_ )

if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument( "fname", help="Input file (as npz)" )
    parser.add_argument( "ofname", help="Output file (as npz)" )
    parser.add_argument( "--seed", default=time.time(), type=long, help="Seed used" )
    parser.add_argument( "--samples", type=float, default=-1, help="Limit number of samples" )

    args = parser.parse_args()

    logger = DataLogger(args.ofname)

    print "Seed:", int( args.seed )
    sc.random.seed( int( args.seed ) )
    logger.add( "seed", int( args.seed ) )

    main( args.fname, int(args.samples) )

