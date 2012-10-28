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

from spectral.rand import permutation

# import spectral.linalg as sl

class GaussianMixtureModel( Model ):
    """Generic mixture model with N components"""
    def __init__( self, prefix, **params ):
        """Create a mixture model for components using given weights"""
        Model.__init__( self, prefix, **params )
        self.k = self.get_parameter( "k" )
        self.d = self.get_parameter( "d" )
        self.weights = self.get_parameter( "w" )
        self.means = self.get_parameter( "M" )
        self.sigmas = self.get_parameter( "S" )

    @staticmethod
    def from_file( prefix ):
        """Load model from a HDF file"""
        model = Model.from_file( prefix ) 
        return GaussianMixtureModel( prefix, **model.params )

    def sample( self, n ):
        """Sample n samples from the model"""

        shape = (n, self.d)

        X = self._allocate_samples( "X", shape )
        # Get a random permutation of N elements
        perm = permutation( n )

        # Sample the number of samples from each view
        cnts = multinomial( n, self.weights )

        cnt_ = 0
        for i in xrange( self.k ):
            cnt = cnts[i]
            # Generate a bunch of points for each mean
            mean, sigma = self.means.T[ i ], self.sigmas[ i ]

            # 1e4 is a decent block size
            def update( start, stop ):
                """Sample random vectors and then assign them to X in
                order"""
                Y = multivariate_normal( mean, sigma, int(stop - start) )
                # Insert into X in a shuffled order
                X[ perm[ start:stop ] ] = Y

            chunked_update( update, cnt_, 10 ** 4, cnt_ + cnt  )
            cnt_ += cnt
        X.flush()
        return X

    @staticmethod
    def generate( prefix, k, d, means = "hypercube", cov = "spherical",
            weights = "random", dirichlet_scale = 10, gaussian_precision
            = 0.01 ):
        """Generate a mixture of k d-dimensional gaussians""" 

        model = Model( prefix )

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

        # Unwrap the store and put it into the appropriate model
        return GaussianMixtureModel( model.prefix, **model.params )

def test_gaussian_mixture_generator_dimensions():
    "Test the GaussianMixtureModel generator"
    import tempfile
    prefix = tempfile.mktemp()

    N = 1000
    D = 100
    K = 3

    gmm = GaussianMixtureModel.generate( prefix, K, D )
    assert( gmm.means.shape == (D, K) )
    assert( gmm.weights.shape == (K,) )

    X = gmm.sample( N )
    # 1000 points with 100 dim each
    assert( X.shape == (N, D) )

    gmm.save()
    gmm_ = GaussianMixtureModel.from_file( prefix )
    Y = gmm_.get_samples( "X", D )
    assert( sc.allclose( X, Y ) )

    gmm.delete()

class MultiViewGaussianMixtureModel( Model ):
    """Generic mixture model with N components"""
    def __init__( self, prefix, **params ):
        Model.__init__( self, prefix, **params )
        self.k = self.get_parameter( "k" )
        self.d = self.get_parameter( "d" )
        self.n_views = self.get_parameter( "v" )
        self.weights = self.get_parameter( "w" )
        self.means = self.get_parameter( "M" )
        self.sigmas = self.get_parameter( "S" )

    @staticmethod
    def from_file( prefix ):
        """Load model from a HDF file"""
        model = Model.from_file( prefix ) 
        return MultiViewGaussianMixtureModel( prefix, **model.params )

    def sample( self, n ):
        """Sample n samples from the mixture model"""

        shape = (n, self.d)

        X = []
        for i in xrange( self.n_views ):
            X.append( self._allocate_samples( "X%d" % (i+1), shape ) )
        # Get a random permutation of N elements
        perm = permutation( n )

        # Sample the number of samples from each view
        cnts = multinomial( n, self.weights )

        # Data for each view
        for view in xrange(self.n_views):
            cnt_ = 0
            for i in xrange( self.k ):
                cnt = cnts[i]
                # Generate a bunch of points for each mean
                mean, sigma = self.means[view].T[ i ], self.sigmas[view][ i ]

                def update( start, stop ):
                    """Sample random vectors and then assign them to X in
                    order"""
                    Y = multivariate_normal( mean, sigma, int(stop - start) )
                    # Insert into X in a shuffled order
                    X[view][ perm[ start:stop ] ] = Y
                chunked_update( update, cnt_, 10 ** 4, cnt_ + cnt  )
                cnt_ += cnt

        return X

    @staticmethod
    def generate( prefix, k, d, n_views = 3, means = "hypercube", cov =
        "spherical", weights = "random", dirichlet_scale = 10,
        gaussian_precision = 0.01 ):
        """Generate a mixture of k d-dimensional multi-view gaussians""" 

        model = Model( prefix )

        model.add_parameter( "k", k )
        model.add_parameter( "d", d )
        model.add_parameter( "v", n_views )

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
            M = []
            for i in xrange( n_views ):
                m = zeros( (d, k) )
                for j in xrange(k):
                    m[(i+j) % k, (i+j) % k] = 1.0
                M.append( m )
            M = array( M )
        elif means == "random":
            M = []
            for i in xrange( n_views ):
                M.append( sc.randn( d, k ) )
            M = array( M )
        elif isinstance( means, sc.ndarray ):
            M = means
        else:
            raise NotImplementedError

        if cov == "spherical":
            # Using 1/gamma instead of inv_gamma
            S = []
            for i in xrange( n_views ):
                sigma = 1/sc.random.gamma(1/gaussian_precision)
                s = array( [ sigma * eye( d ) for i in xrange( k ) ] )
                S.append( s )
            S = array( S ) 
        elif isinstance( cov, sc.ndarray ):
            S = cov
        else:
            # TODO: Implement random wishart and other variations.
            raise NotImplementedError

        model.add_parameter( "w", w )
        model.add_parameter( "M", M )
        model.add_parameter( "S", S )

        # Unwrap the store and put it into the appropriate model
        return MultiViewGaussianMixtureModel( model.prefix, **model.params )

def test_mv_gaussian_mixture_generator_dimensions():
    "Test the MultiViewGaussianMixtureModel generator"
    import tempfile
    prefix = tempfile.mktemp()

    N = 1000
    D = 100
    K = 3
    VIEWS = 3

    mvgmm = MultiViewGaussianMixtureModel.generate( prefix, K, D, n_views = VIEWS )
    assert( len( mvgmm.means ) == K )
    assert( len( mvgmm.sigmas ) == K )
    for view in xrange(VIEWS):
        assert( mvgmm.means[view].shape == (D, K) )
    assert( mvgmm.weights.shape == (K,) )

    points = mvgmm.sample( N )
    # 1000 points with 100 dim each
    assert( len( points ) == K )
    for view in xrange(VIEWS):
        assert( points[view].shape == (N, D) )

    mvgmm.save()
    mvgmm.delete()

