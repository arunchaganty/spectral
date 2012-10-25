"""
E-M algorithm to detect and separate GMMs
"""

#import ipdb
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
        closest_permuted_vector, column_aerr, column_rerr
from spectral.util import DataLogger

logger = DataLogger("log")

class GaussianMixtureEM( em.EMAlgorithm ):
    """
    Gaussian Mixtures EM
    (i) Using k-means++ start
    (ii) Assuming spherical gaussians
    """

    def __init__( self, k, d ):
        self.k, self.d = k, d
        em.EMAlgorithm.__init__( self )

    def compute_expectation( self, X, O ):
        """Compute the most likely values of the latent variables; returns lhood"""
        N, d = X.shape
        M, sigma, w = O

        total_lhood = 0
        # Get pairwise distances between centers (D_ij = \|X_i - M_j\|)
        D = cdist( X, M.T )
        # Probability dist = 1/2(\sigma^2) D^2 + log w
        Z = - 0.5/sigma**2 * (D**2) + log( w ) - 0.5 * d * log(sigma) # Ignoreing constant term
        total_lhood += logsumexp( logsumexp(Z) )

        # Normalise the probilities (soft EM)
        Z = sc.exp(Z.T - logsumexp(Z,1)).T
            
        return total_lhood, Z

    def compute_maximisation( self, X, Z, O ):
        """Compute the most likely values of the parameters"""
        N, d = X.shape

        M, sigma, w = O

        # Cluster weights (smoothed)
        # Pseudo counts
        w = Z.sum(axis=0) + 1

        # Get new means
        M = (Z.T.dot( X ).T / w)
        sigma = (cdist( X, M.T ) * Z).sum()/(d*N)
        w /= w.sum()

        return M, sigma, w

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
            O = self.kmeanspp_initialisation( X )
        return em.EMAlgorithm.run( self, X, O, *args, **kwargs )

def test_gaussian_em():
    """Test the Gaussian EM on a small generated dataset"""

    k, d = 2, 2

    M1 = array( [ -1.0, 1.0 ] ) # Diagonally placed centers
    M2 = array( [ 1.0, -1.0 ] )
    M = sc.row_stack( (M1,M2) )
    sigma = 0.2
    w = array( [0.5, 0.5] )

    X1 = multivariate_normal( M1, sigma*eye(2), 10000 )
    X2 = multivariate_normal( M2, sigma*eye(2), 10000 )
    X = sc.row_stack( (X1, X2) )

    algo = GaussianMixtureEM(k, d)

    lhood, Z, O = algo.run( X )
    M_, sigma_, w_ = O

    M_ = closest_permuted_matrix( M, M_ )
    w_ = closest_permuted_vector( w, w_ )

    print norm( M - M_ )/norm(M)
    print abs(sigma - sigma_) 
    print norm( w - w_ ) 

    assert( norm( M - M_ )/norm(M) < 1e-1 )
    assert( abs(sigma - sigma_) < 1 )
    assert( norm( w - w_ ) < 1e-3 )

def main(fname, samples):
    """Run GMM EM on the data in @fname"""

    gmm = sc.load( fname )
    k, d, M, S, w, X = gmm['k'], gmm['d'], gmm['M'], gmm['S'], gmm['w'], gmm['X']

    algo = GaussianMixtureEM( k, d )

    N, _ = X.shape

    if (samples < 0 or samples > N):
        print "Warning: %s greater than number of samples in file. Using %s instead." % ( samples, N )
    else:
        X = X[:samples, :]
    N, _ = X.shape

    logger.add( "k", k )
    logger.add( "d", d )
    logger.add( "N", N )
    logger.add( "M", M )

    lhood, Z, O = algo.run( X )
    M_, S_, w_ = O

    M_ = closest_permuted_matrix( M.T, M_.T ).T
    logger.add( "M_", M_ )

    # Table
    logger.add_err( "M", M, M_ )
    logger.add_err( "M", M, M_, 'col' )
    print column_aerr( M, M_ ), column_rerr( M, M_ )

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

