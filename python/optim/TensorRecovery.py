"""
Phase recovery of a tensor using proximal subgradient descent
"""

import scipy as sc
from scipy import diag, array, outer, eye, ones, sqrt, zeros, einsum
from scipy.linalg import norm, svd, svdvals 
from spectral.linalg import tensorify, HOSVD, fold, unfold
from spectral.data import Txyz

from optim.ProximalGradient import ProximalGradient

class TensorRecovery( ProximalGradient ):
    """
    Solve the problem:
    argmin_{B} ( B(x,x,x) - y ) + \lambda \| B_i \|_*
    """
    def initialise( self, y, X ):
        _, d = X.shape
        return zeros( (d, d, d) )

    def gradient_step( self, y, X, B ):
        """
        Take a gradient step along each B to optimise 
        $\diff{L}{B_{ij}} = \sum_i (B(x^i, x^i, x^i)' - y^i^2) x^i \otimes x^i \otimes x^i 
        """ 

        d = B.shape[0]
        N = len(y)

        # Compute x^T B x - y
        dB = zeros( (d, d, d) )
        Z = (Txyz( B, X, X, X ) - y)

        for i in xrange( N ):
            x_i = X[i]
            dB += Z[i] * tensorify( x_i, x_i, x_i )

        return dB/N

    def proximal_step( self, y, X, B, thresh ):
        """Make X low rank"""
        # Threshold the singular values of each mode of the tensor
        for i in xrange( B.ndim ):
            B_i = unfold( B, i+1 )
            U, S, Vt = svd( B_i, full_matrices = False )
            S -= thresh
            S[ S < 0.0 ] = 0
            B = fold( U.dot( diag( S ) ).dot( Vt ), i+1, B.shape )
        return B

    def loss( self, y, X, B ):
        """Residual in error"""
        N = len( y )
        tot = 0
        for (x_i, y_i) in zip( X, y ):
            tot += (einsum( "ijk,i,j,k", B, x_i, x_i, x_i) - y_i)**2
        return tot/N

    def solve( self, y, X, *args, **kwargs ):
        """Solve using a Q value"""

        return ProximalGradient.solve( self, y, X, *args, **kwargs )

def test_solve_exact(samples = 1e4, iters = 1e2):
    K = 2
    d = 3
    N = samples

    pi = array( [0.5, 0.5] ) 
    B = eye( d, K )
    B3 = sum( [ pi[i] * tensorify(B.T[i], B.T[i], B.T[i] ) for i in xrange(K) ] )

    X = sc.randn( N, d )
    y = array( [ einsum( "ijk,i,j,k", B3, x, x, x ) for x in X ] )

    algo = TensorRecovery()
    B3_ = algo.solve( y, X, iters = iters, alpha0 = 0.01, eps=1e-5, reg = 1e-3 )

    _, _, V = HOSVD( B3 )
    print algo.loss( y, X, B3 ), V

    __, _, V_ = HOSVD( B3_ )
    print algo.loss( y, X, B3_ ), V_
    print norm(B3 - B3_)/norm(B3)
    
    assert norm(B3 - B3_)/norm(B3) < 1e-1

def test_solve_samples(samples, iters):
    from models import LinearRegressionsMixture
    K = 2
    d = 3
    N = samples

    fname = "/tmp/test.npz"
    lrm = LinearRegressionsMixture.generate(fname, K, d, cov = "eye", betas="random")
    M, S, pi, B = lrm.mean, lrm.sigma, lrm.weights, lrm.betas
    B3 = sum( [ pi[i] * tensorify(B.T[i], B.T[i], B.T[i] ) for i in xrange(K) ] )

    y, X = lrm.sample( N )
    y = y**3

    algo = TensorRecovery()
    B3_ = algo.solve( y, X, iters = iters, alpha0 = 0.01, eps=1e-5, reg = 1e-3 )

    _, _, V = HOSVD( B3 )
    print algo.loss( y, X, B3 ), V

    __, _, V_ = HOSVD( B3_ )
    print algo.loss( y, X, B3_ ), V_
    print norm(B3 - B3_)/norm(B3)
    
    assert norm(B3 - B3_)/norm(B3) < 1e-1

if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument( "--samples", default=1e6, type=float, help="Number of samples to be used" )
    parser.add_argument( "--iters", default=1e2, type=float, help="Number of iterations to be used" )
    parser.add_argument( "--with-noise", default=False, type=bool, help="Use noise" )

    args = parser.parse_args()

    test_solve_exact(int(args.samples), int( args.iters) )
    test_solve_samples(int(args.samples), int( args.iters) )


