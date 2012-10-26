"""
Generate datasets
"""

import scipy as sc 
from scipy import array, eye
from models import GaussianMixtureModel

def main( fname, dataset_type, N, k, d, params ):
    """Generate dataset in file fname"""
    if dataset_type == "gmm":
        if params.cov == "spherical" and params.sigma2 > 0:
            params.cov = array( [params.sigma2 * eye(d)] * k )
        gmm = GaussianMixtureModel.generate( fname, k, d, params.means,
                params.cov, params.weights )
        gmm.sample( N )

        gmm.close() 
    #elif dataset_type == "mvgmm":
    #    views = params.views 
    #    if params.cov == "spherical" and params.sigma2 > 0:
    #        params.cov = array( [params.sigma2 * eye(d)] * views  )
    #    mvgmm = MultiViewGaussianMixtureModel.generate( fname, k, d, views, params.means,
    #            params.cov, params.weights )
    #    mvgmm.sample( N )
    #    mvgmm.save()
    else:
        raise NotImplementedError
                
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "fname", help="Output file (as hdf)" )
    parser.add_argument( "model", help="Model: gmm|mvgmm|lda" )
    parser.add_argument( "N", type=float, help="Number of samples" )
    parser.add_argument( "k", type=int, help="Number of mixture components"  )
    parser.add_argument( "d", type=int, help="Dimensionality of each component"  )

    parser.add_argument( "--weights", default="uniform" )
    # GMM options
    parser.add_argument( "--means", default="hypercube" )
    parser.add_argument( "--cov", default="spherical" )
    parser.add_argument( "--sigma2", default=0.2, type=float )
    # LDA options
    parser.add_argument( "--a0", default=10, type=float )
    parser.add_argument( "--wpd", default=100, help="Words per document", type=int )
    # Multiview options
    parser.add_argument( "--views", default=3, help="Number of views", type=int )

    args = parser.parse_args()

    main( args.fname, args.model, int(args.N), args.k, args.d, args )

