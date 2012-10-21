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
logsumexp = scipy.misc.logsumexp

from spectral.linalg import closest_permuted_matrix

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
        total_lhood += logsumexp(Z)

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

        sigma = cdist( X, M ).sum()/(k*d*N)
        w = ones( k )/float(k)

        return sc.column_stack(M), sigma, w

    def run( self, X, O = None, *args, **kwargs ):
        if O == None:
            X1, X2, X3 = X
            M1, S1, w = self.kmeanspp_initialisation( X1 )
            M2, S2, _ = self.kmeanspp_initialisation( X2 )
            M3, S3, _ = self.kmeanspp_initialisation( X3 )
            O = (M1, M2, M3), (S1, S2, S3), w
        return em.EMAlgorithm.run( self, X, O, *args, **kwargs )

def test_multiview_gmm_em():
    mvgmm = sc.load( "test-data/mvgmm-2-3-1e4.npz" )
    k, d, M, S, w, X = mvgmm['k'], mvgmm['d'], mvgmm['M'], mvgmm['S'], mvgmm['w'], mvgmm['X']

    algo = MultiViewGaussianMixtureEM( k, d )

    lhood, Z, O = algo.run( X )
    (M1_, M2_, M3_), (S1, S2, S3), w = O

    M1, M2, M3 = M

    M1_ = closest_permuted_matrix( M1, M1_ )
    M2_ = closest_permuted_matrix( M2, M2_ )
    M3_ = closest_permuted_matrix( M3, M3_ )

    assert norm(M1 - M1_)/norm(M1) < 1e-2
    assert norm(M2 - M2_)/norm(M2) < 1e-2
    assert norm(M3 - M3_)/norm(M3) < 1e-2


def main(fname):
    """Run MVGMM EM on the data in @fname"""

    mvgmm = sc.load( fname )
    k, d, M, S, w, X = mvgmm['k'], mvgmm['d'], mvgmm['M'], mvgmm['S'], mvgmm['w'], mvgmm['X']

    algo = MultiViewGaussianMixtureEM( k, d )

    lhood, Z, O = algo.run( X )
    (M1_, M2_, M3_), (S1, S2, S3), w = O

    M1, M2, M3 = M

    M1_ = closest_permuted_matrix( M1, M1_ )
    M2_ = closest_permuted_matrix( M2, M2_ )
    M3_ = closest_permuted_matrix( M3, M3_ )

    print "Error in M1: ", (norm(M1 - M1_)/norm(M1))
    print "Error in M2: ", (norm(M2 - M2_)/norm(M2))
    print "Error in M3: ", (norm(M3 - M3_)/norm(M3))

if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument( "fname", help="Input file (as npz)" )
    parser.add_argument( "--seed", default=time.time(), type=long, help="Input file (as npz)" )

    args = parser.parse_args()
    print "Seed:", int( args.seed )
    sc.random.seed( int( args.seed ) )


    main( args.fname )

