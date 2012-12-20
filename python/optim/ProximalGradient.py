"""
General proximal gradient methods
"""

import scipy as sc
from scipy import diag, array, eye, ones, sqrt, zeros
from scipy.linalg import norm 
from util import ErrorBar

class ProximalGradient:
    """A proximal gradient method consists of two steps:
    a convex gradient step and a proximal step
    """
    def __init__(self):
        pass

    def initialise( self, y, X ):
        """Find a suitable initialisation point"""
        raise NotImplementedError()

    def gradient_step( self, y, X, B ):
        """Perform a single gradient step"""
        raise NotImplementedError()

    def proximal_step( self, y, X, B, reg ):
        """Perform a single proximal step"""
        raise NotImplementedError()

    def loss( self, y, X, B ):
        """Compute loss for y and X"""
        raise NotImplementedError()


    def solve( self, y, X, B0 = None, reg = 1e-2, iters = 500, eps = 1e-4, alpha0 = 0.1, alpha = "1/T", verbose = True ):
        """
        Solve the problem B = argmin L( f(X;B), y ) + \|B\|
        """

        # Initialise the B0
        if B0 is None:
            B0 = self.initialise( y, X )

        if verbose:
            ebar = ErrorBar()
            ebar.start( iters )

        B_ = B0
        for i in xrange( iters ):
            # Run a gradient descent step
            dB = self.gradient_step( y, X, B_ )

            # Anneal
            if alpha == "1/T" :
                alpha = alpha0/(i+1)
            elif alpha == "1/sqrt(T)":
                alpha = alpha0/sqrt(i+1)
            B = B_ - alpha * dB

            # Do the proximal step of making B low rank by soft thresholding k.
            B = self.proximal_step( y, X, B, reg )

            if verbose:
                ebar.update( i, self.loss( y, X, B ) )

            # Check convergence
            if norm( B - B_ ) / (1e-10 + norm( B )) < eps:
                break
            B_ = B

        if verbose:
            ebar.stop()

        return B

