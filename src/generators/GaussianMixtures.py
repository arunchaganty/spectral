"""
Generate data from a Gaussian mixture model
"""

import scipy as sc
from scipy import matrix, array
from scipy.linalg import norm 
from generators.MixtureModel import MixtureModel

multinomial = sc.random.multinomial
multivariate_normal = sc.random.multivariate_normal 
dirichlet = sc.random.dirichlet

import spectral.rand as sr
import spectral.linalg as sl

class GaussianMixtureModel( MixtureModel ):
    """Generic mixture model with N components"""
    def __init__( self, weights, means, sigmas ):
        """Create a mixture model for components using given weights"""
        self.K = len( weights )
        MixtureModel.__init__(self, weights, means)
        self.means = means
        self.sigmas = sigmas

    def sample( self, n ):
        """Sample n samples from the mixture model"""
        # Sample the number of samples from each view
        cnts = multinomial( n, self.weights )

        # Shuffle all of the data 
        # Row permutation only
        #shuffle = sr.permutation( n ), None

        points = []
        for (k, cnt) in zip( xrange( self.K ), cnts ):
            # Generate a bunch of points for each mean
            mean, sigma = self.means.T[ k ], self.sigmas[ k ]
            points.append( multivariate_normal( mean, sigma, cnt  ) )
        points = sc.row_stack( points )

        # Permute the rows
        #points = sl.apply_matrix_permutation( shuffle, points )
        return points

    @staticmethod
    def generate( k, d, dirichlet_scale = 10, gaussian_precision = 0.01, cov = "spherical" ):
        """Generate a mixture of k d-dimensional gaussians""" 

        if cov == "spherical":
            means = sc.randn( d, k )
            # Using 1/gamma instead of inv_gamma
            sigma = 1/sc.random.gamma(1/gaussian_precision)
            sigmas = [ sigma * sc.eye( d ) for i in xrange( k ) ]
        else:
            # TODO: Implement random wishart and other variations.
            raise NotImplementedError

        weights = dirichlet( sc.ones(k) * dirichlet_scale ) 
        return GaussianMixtureModel( weights, means, sigmas )

def test_gaussian_mixture_generator_dimensions():
    "Test the GaussianMixtureModel generator"
    N = 1000
    D = 100
    K = 3

    gmm = GaussianMixtureModel.generate( K, D )
    assert( gmm.means.shape == (D, K) )
    assert( gmm.weights.shape == (K,) )

    points = gmm.sample( N )
    # 1000 points with 100 dim each
    assert( points.shape == (N, D) )

class MultiViewGaussianMixtureModel( MixtureModel ):
    """Generic mixture model with N components"""
    def __init__( self, weights, means, sigmas ):
        """Create a mixture model for components using given weights"""

        MixtureModel.__init__(self, weights, means)
        self.K = len(weights)
        self.n_views = len( means )
        self.means = means
        self.sigmas = sigmas

    def sample( self, n ):
        """Sample n samples from the mixture model"""

        # Sample the number of samples from each view
        cnts = multinomial( n, self.weights )

        # Shuffle all of the data 
        # Row permutation only
        #shuffle = sr.permutation( n ), None

        # Data for each view
        points = []
        for view in xrange(self.n_views):
            points_ = []

            for (k, cnt) in zip( xrange( self.K ), cnts ):
                # Generate a bunch of points for each mean
                mean, sigma = self.means[view].T[ k ], self.sigmas[view][ k ]
                points_.append( multivariate_normal( mean, sigma, cnt  ) )
            points_ = sc.row_stack( points_ )
            #points_ = sl.apply_matrix_permutation( shuffle, points_ )

            points.append( points_ )

        return points

    @staticmethod
    def generate( k, d, n_views = 3, dirichlet_scale = 10, gaussian_precision = 0.01, cov = "spherical" ):
        """Generate a mixture of k d-dimensional multi-view gaussians""" 

        if cov == "spherical":
            means, sigmas = [], []
            for i in xrange( n_views ):
                means.append( sc.randn( d, k ) )

                # Each view could potentially have a different
                # covariance
                # Using 1/gamma instead of inv_gamma
                sigma = 1/sc.random.gamma(1/gaussian_precision)
                sigmas.append( [ sigma * sc.eye( d ) for i in xrange( k ) ] )
        else:
            # TODO: Implement random wishart and other variations.
            raise NotImplementedError

        weights = dirichlet( sc.ones(k) * dirichlet_scale ) 
        return MultiViewGaussianMixtureModel( weights, means, sigmas )

def test_mv_gaussian_mixture_generator_dimensions():
    "Test the MultiViewGaussianMixtureModel generator"

    N = 1000
    D = 100
    K = 3
    VIEWS = 3

    gmm = MultiViewGaussianMixtureModel.generate( K, D, n_views = VIEWS )
    assert( len( gmm.means ) == K )
    assert( len( gmm.sigmas ) == K )
    for view in xrange(VIEWS):
        assert( gmm.means[view].shape == (D, K) )
    assert( gmm.weights.shape == (K,) )

    points = gmm.sample( N )
    # 1000 points with 100 dim each
    assert( len( points ) == K )
    for view in xrange(VIEWS):
        assert( points[view].shape == (N, D) )

