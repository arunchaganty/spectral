"""
Phase recovery using proximal subgradient descent
"""

import scipy as sc
from scipy import diag, array, ndim, outer, eye, ones, sqrt, zeros
from scipy.linalg import norm, svd, svdvals 

from spectral.data import xMy

from optim.ProximalGradient import ProximalGradient

class PhaseRecovery( ProximalGradient ):
    """
    Solve the problem:
    argmin_{B} ( x^T B x - y ) + \lambda \| B \|_*
    """

    def initialise( self, y, X ):
        _, d = X.shape
        return zeros( (d, d) )

    def gradient_step( self, y, X, B ):
        """
        Take a gradient step along each B to optimise 
        $\diff{L}{B_{ij}} = \sum_i (x^i' B x^i - y^i^2) x^i_i x^i_j
        """ 

        Q2 = self.Q2
        d, _ = B.shape
        N = len(y)

        assert( Q2.shape == (N,N) )

        # Compute x^T B x - y
        dB = zeros( (d, d) )
        # Forgive me father for I have multiplied two large matrices.
        Z = ( xMy( B, X, X ) - y).dot( Q2 )
        assert( Z.shape == (N,) )

        for i in xrange( N ):
            x_i = X[i]
            dB += Z[i] * outer( x_i, x_i )

        return dB/N

    def proximal_step( self, y, X, B, thresh ):
        """Make X low rank"""
        U, S, Vt = svd( B, full_matrices = False )
        S -= thresh
        S[ S < 0.0 ] = 0.0
        return U.dot( diag( S ) ).dot( Vt )

    def loss( self, y, X, B ):
        """Residual in error"""
        N = len( y )
        tot = 0
        for (x_i, y_i) in zip( X, y ):
            tot += (x_i.dot( B ).dot( x_i ) - y_i)**2
        return tot/N

    def solve( self, y, X, Q = None, *args, **kwargs ):
        """Solve using a Q value"""
        N = len(y)

        if Q is None:
            Q = eye( N )
        self.Q = Q
        self.Q2 = Q.dot( Q.T )

        return ProximalGradient.solve( self, y, X, *args, **kwargs )


def test_solve_exact(samples = 1e4, iters = 100):
    N = samples
    d = 3
    k = 2

    pi = array( [0.5, 0.5] ) 
    B = sc.randn( d, k )
    B2 = B.dot( diag( pi ) ).dot( B.T )

    X = sc.randn( N, d )
    y = array( [ x.T.dot( B2 ).dot( x ) for x in X ] )

    algo = PhaseRecovery()
    B2_ = algo.solve( y, X, iters = iters )

    print algo.loss( y, X, B2 )
    print algo.loss( y, X, B2_ )

    print norm(B2 - B2_)/norm(B2) 

    assert norm(B2 - B2_)/norm(B2) < 1e-1

def test_solve_samples(samples = 1e4, iters = 100):
    from models import LinearRegressionsMixture
    K = 2
    d = 3
    N = samples

    fname = "/tmp/test.npz"
    lrm = LinearRegressionsMixture.generate(fname, K, d, cov = "eye", betas="random")
    M, S, pi, B = lrm.mean, lrm.sigma, lrm.weights, lrm.betas
    B2 = B.dot( diag( pi ) ).dot( B.T )

    y, X = lrm.sample( N )
    y = y**2

    algo = PhaseRecovery()
    B2_ = algo.solve( y, X, iters = iters )

    print "Exact:", algo.loss( y, X, B2 ), svdvals(B2)
    print "Recovered:", algo.loss( y, X, B2_ ), svdvals(B2_)
    print norm(B2 - B2_)/norm(B2) 

    assert norm(B2 - B2_)/norm(B2) < 1e-1

if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument( "--samples", default=1e4, type=float, help="Number of samples to be used" )
    parser.add_argument( "--iters", default=1e2, type=float, help="Number of iterations to be used" )
    parser.add_argument( "--with-noise", default=False, type=bool, help="Use noise" )

    args = parser.parse_args()

    test_solve_exact(int(args.samples), int( args.iters) )
    test_solve_samples(int(args.samples), int( args.iters) )

