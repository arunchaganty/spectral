"""
The EM Algorithm
"""

import time

class EMAlgorithm:
    """The expectation maximisation algorithm. Derivers are expected to
    fill in the expectation and maximisation steps"""
    def __init__( self ):
        pass

    def compute_expectation( self, X, O ):
        """Compute the most likely values of the latent variables; returns lhood"""

        raise NotImplementedError

    def compute_maximisation( self, X, Z, O ):
        """Compute the most likely values of the parameters"""

        raise NotImplementedError

    def run( self, X, O, iter_cb = None, iters=100, eps=1e-5 ):
        """Run with some initial values of parameters O"""

        lhood, Z = self.compute_expectation(X, O)
        for i in xrange( iters ):
            print "Iteration %d, lhood = %f" % (i, lhood)
            O = self.compute_maximisation(X, Z, O)
            # Add error and time to log
            if iter_cb is not None:
                iter_cb( i, O, lhood )

            lhood_, Z = self.compute_expectation(X, O)
            if abs(lhood_ - lhood) < eps:
                print "Converged with lhood=%f in %d steps." % ( lhood, i )
                lhood = lhood_
                break
            else:
                lhood = lhood_

        return lhood, Z, O

