"""
Generate data from a Gaussian mixture model
"""

import scipy as sc
from scipy import array, zeros, ones, eye
from models.Model import Model
from util import chunked_update #, ProgressBar

multinomial = sc.random.multinomial
multivariate_normal = sc.random.multivariate_normal 
dirichlet = sc.random.dirichlet

# import spectral.linalg as sl

class GaussianMixtureModel( Model ):
    """Generic mixture model with N components"""
    def __init__( self, store ):
        """Create a mixture model for components using given weights"""
        Model.__init__( self, store )
        self.k = self.get_parameter( "k" )
        self.d = self.get_parameter( "d" )
        self.weights = self.get_parameter( "w" )
        self.means = self.get_parameter( "M" )
        self.sigmas = self.get_parameter( "S" )

    def sample( self, n ):
        """Sample n samples from the model"""

        shape = (n, self.d)

        X = self._allocate_samples( "X", shape )

        # Sample the number of samples from each view
        cnts = multinomial( n, self.weights )

        cnt_ = 0
        for i in xrange( self.k ):
            cnt = cnts[i]
            # Generate a bunch of points for each mean
            mean, sigma = self.means.T[ i ], self.sigmas[ i ]
            # 1e4 is a decent block size
            chunked_update( X[ cnt_ : cnt_ + cnt ], cnt, 10**4,
                    multivariate_normal, mean, sigma ) 

        return X

    @staticmethod
    def generate( fname, k, d, means = "hypercube", cov = "spherical",
            weights = "random", dirichlet_scale = 10, gaussian_precision
            = 0.01 ):
        """Generate a mixture of k d-dimensional gaussians""" 

        model = Model.create( fname )

        model.add_parameter( "k", k )
        model.add_parameter( "d", d )

        if weights == "random":
            w = dirichlet( ones(k) * dirichlet_scale ) 
        elif weights == "uniform":
            w = ones(k)/k
        elif isinstance( weights, sc.ndarray ):
            w = weights
        else:
            raise NotImplementedError

        if means == "hypercube":
            # Place means at the vertices of the hypercube
            M = zeros( (d, k) )
            for i in xrange(k):
                M[i, i] = 1.0
        elif means == "random":
            M = sc.randn( d, k )
        elif isinstance( means, sc.ndarray ):
            M = means
        else:
            raise NotImplementedError

        if cov == "spherical":
            # Using 1/gamma instead of inv_gamma
            sigma = 1/sc.random.gamma(1/gaussian_precision)
            S = array( [ sigma * eye( d ) for i in xrange( k ) ] )
        elif isinstance( cov, sc.ndarray ):
            S = cov
        else:
            # TODO: Implement random wishart and other variations.
            raise NotImplementedError

        model.add_parameter( "w", w )
        model.add_parameter( "M", M )
        model.add_parameter( "S", S )

        # Unwrap the store and put it into a GaussianMixtureModel
        return GaussianMixtureModel( model.store )

def test_gaussian_mixture_generator_dimensions():
    "Test the GaussianMixtureModel generator"
    import os, tempfile
    fname = tempfile.mktemp()

    N = 1000
    D = 100
    K = 3

    gmm = GaussianMixtureModel.generate( fname, K, D )
    assert( gmm.means.shape == (D, K) )
    assert( gmm.weights.shape == (K,) )

    points = gmm.sample( N )
    # 1000 points with 100 dim each
    assert( points.shape == (N, D) )

    gmm.close()

    os.remove( fname )

#class MultiViewGaussianMixtureModel( MixtureModel ):
#    """Generic mixture model with N components"""
#    def __init__( self, weights, means, sigmas ):
#        """Create a mixture model for components using given weights"""
#
#        MixtureModel.__init__(self, weights, means)
#        self.K = len(weights)
#        self.n_views = len( means )
#        self.means = means
#        self.sigmas = sigmas
#
#    def sample( self, n ):
#        """Sample n samples from the mixture model"""
#
#        # Sample the number of samples from each view
#        cnts = multinomial( n, self.weights )
#
#        # Shuffle all of the data 
#        # Row permutation only
#        #shuffle = sr.permutation( n ), None
#
#        # Data for each view
#        points = []
#        for view in xrange(self.n_views):
#            points_ = []
#
#            for (k, cnt) in zip( xrange( self.K ), cnts ):
#                # Generate a bunch of points for each mean
#                mean, sigma = self.means[view].T[ k ], self.sigmas[view][ k ]
#                points_.append( multivariate_normal( mean, sigma, cnt  ) )
#            points_ = sc.row_stack( points_ )
#            #points_ = sl.apply_matrix_permutation( shuffle, points_ )
#
#            points.append( points_ )
#
#        return points
#
#    @staticmethod
#    def generate( k, d, n_views = 3, dirichlet_scale = 10, gaussian_precision = 0.01, cov = "spherical" ):
#        """Generate a mixture of k d-dimensional multi-view gaussians""" 
#
#        if cov == "spherical":
#            means, sigmas = [], []
#            for i in xrange( n_views ):
#                means.append( sc.randn( d, k ) )
#
#                # Each view could potentially have a different
#                # covariance
#                # Using 1/gamma instead of inv_gamma
#                sigma = 1/sc.random.gamma(1/gaussian_precision)
#                sigmas.append( [ sigma * sc.eye( d ) for i in xrange( k ) ] )
#        else:
#            # TODO: Implement random wishart and other variations.
#            raise NotImplementedError
#
#        weights = dirichlet( ones(k) * dirichlet_scale ) 
#        return MultiViewGaussianMixtureModel( weights, means, sigmas )
#
#def test_mv_gaussian_mixture_generator_dimensions():
#    "Test the MultiViewGaussianMixtureModel generator"
#
#    N = 1000
#    D = 100
#    K = 3
#    VIEWS = 3
#
#    gmm = MultiViewGaussianMixtureModel.generate( K, D, n_views = VIEWS )
#    assert( len( gmm.means ) == K )
#    assert( len( gmm.sigmas ) == K )
#    for view in xrange(VIEWS):
#        assert( gmm.means[view].shape == (D, K) )
#    assert( gmm.weights.shape == (K,) )
#
#    points = gmm.sample( N )
#    # 1000 points with 100 dim each
#    assert( len( points ) == K )
#    for view in xrange(VIEWS):
#        assert( points[view].shape == (N, D) )
#
