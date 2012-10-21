"""
E-M algorithm to detect and separate GMMs
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
logsumexp = scipy.misc.logsumexp

from spectral.linalg import closest_permuted_matrix, closest_permuted_vector

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
        D = cdist( X, M )
        # Probability dist = 1/2(\sigma^2) D^2 + log w
        Z = - 0.5/sigma**2 * (D**2) + log( w ) - 0.5 * d * log(sigma) # Ignoreing constant term
        total_lhood += logsumexp(Z)

        # Normalise the probilities (soft EM)
        Z = sc.exp(Z.T - logsumexp(Z,1)).T
            
        return total_lhood, Z

    def compute_maximisation( self, X, Z, O ):
        """Compute the most likely values of the parameters"""
        N, d = X.shape

        M, sigma, w = O

        # Cluster weights (smoothed)
        # Pseudo counts
        P = Z.sum(axis=0) + 1

        # Get new means
        M = (Z.T.dot( X ).T / P).T
        sigma = (cdist( X, M ) * Z).sum()/(d*N)
        w = P/P.sum()

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

        sigma = cdist( X, M ).sum()/(k*d*N)
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

    X1 = multivariate_normal( M1, sigma*eye(2), 1000 )
    X2 = multivariate_normal( M2, sigma*eye(2), 1000 )
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

