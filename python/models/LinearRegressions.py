"""
Generate data from a mixture of linear regressions
"""

import scipy as sc
import scipy.linalg
from scipy import array, zeros, ones, eye
from scipy.linalg import inv
from models.Model import Model
from util import chunked_update #, ProgressBar

multinomial = sc.random.multinomial
multivariate_normal = sc.random.multivariate_normal 
dirichlet = sc.random.dirichlet

from spectral.rand import permutation, wishart

# import spectral.linalg as sl

class LinearRegressionsMixture( Model ):
    """Generic mixture model with N components"""
    def __init__( self, fname, **params ):
        """Create a mixture model for components using given weights"""
        Model.__init__( self, fname, **params )
        self.k = self.get_parameter( "k" )
        self.d = self.get_parameter( "d" )

        self.weights = self.get_parameter( "w" )
        self.betas = self.get_parameter( "B" )

        self.mean = self.get_parameter( "M" )
        self.sigma = self.get_parameter( "S" )

    @staticmethod
    def from_file( fname ):
        """Load model from a HDF file"""
        model = Model.from_file( fname ) 
        return LinearRegressionsMixture( fname, **model.params )

    def sample( self, N, n = -1 ):
        """Sample N samples from the model. If N, n are both specified,
        then generate N samples, but only keep n of them"""

        raise NotImplementedError

    @staticmethod
    def generate( fname, k, d, mean = "zero", cov = "random", betas = "random", weights = "random",
            dirichlet_scale = 10, gaussian_precision = 0.01 ):
        """Generate a mixture of k d-dimensional multi-view gaussians""" 

        model = Model( fname )
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

        elif betas == "random":
            B = sc.randn( d )
        elif isinstance( betas, sc.ndarray ):
            B = betas
        else:
            raise NotImplementedError

        if means == "zero":
            M = zeros( d )
        elif means == "random":
            M = sc.randn( d )
        elif isinstance( means, sc.ndarray ):
            M = means
        else:
            raise NotImplementedError

        if cov == "spherical":
            # Using 1/gamma instead of inv_gamma
            sigma = 1/sc.random.gamma(1/gaussian_precision)
            S = sigma * eye( d )
        elif cov == "random":
            S = gaussian_precision * inv( wishart( d+1, sc.eye( d ), 1 ) ) 
        elif isinstance( cov, sc.ndarray ):
            S = cov
        else:
            raise NotImplementedError

        model.add_parameter( "w", w )
        model.add_parameter( "B", M )
        model.add_parameter( "M", M )
        model.add_parameter( "S", S )

        # Unwrap the store and put it into the appropriate model
        return LinearRegressionsMixture( model.fname, **model.params )



if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument( "fname", help="Output file (as npz)" )
    parser.add_argument( "model", help="Model: gmm|mvgmm" )
    parser.add_argument( "k", type=int, help="Number of mixture components"  )
    parser.add_argument( "d", type=int, help="Dimensionality of each component"  )

    parser.add_argument( "--seed", default=int(time.time() * 1000), type=int )
    parser.add_argument( "--weights", default="uniform", help="Mixture weights, default=uniform|random" )
    parser.add_argument( "--w0", default=10.0, type=float, help="Scale parameter for the Dirichlet" )
    # GMM options
    parser.add_argument( "--means", default="hypercube", help="Mean generation procedure, default = hypercube" )
    parser.add_argument( "--cov", default="spherical", help="Covariance generation procedure, default = spherical|random"  )
    parser.add_argument( "--sigma2", default=0.2, type=float )
    # Multiview options
    parser.add_argument( "--views", default=3, help="Number of views", type=int )

    args = parser.parse_args()
    sc.random.seed( int( args.seed ) )

    main( args.fname, args.model, args.k, args.d, args )




