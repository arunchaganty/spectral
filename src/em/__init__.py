"""
The EM Algorithm
"""

import time

class EMAlgorithm:
    """The expectation maximisation algorithm. Derivers are expected to
    fill in the expectation and maximisation steps"""
    def __init__( self, logger = None ):
        self.logger = logger

    def compute_expectation( self, X, O ):
        """Compute the most likely values of the latent variables; returns lhood"""

        raise NotImplementedError

    def compute_maximisation( self, X, Z, O ):
        """Compute the most likely values of the parameters"""

        raise NotImplementedError

    def run( self, X, O, O_, iters=100, eps=1e-5 ):
        """Run with some initial values of parameters O_; O is the true values"""

        start = time.time()

        lhood, Z = self.compute_expectation(X, O_)
        for i in xrange( iters ):
            print "Iteration %d, lhood = %f" % (i, lhood)
            O_ = self.compute_maximisation(X, Z, O_)
            # Add error and time to log
            if self.logger:
                self.logger.add_err( "M_%d" % i, O, O_ )
                self.logger.add_err( "time_%d" % i, (time.time() - start) )

            lhood_, Z = self.compute_expectation(X, O_)
            if abs(lhood_ - lhood) < eps:
                print "Converged with lhood=%f in %d steps." % ( lhood, i )
                lhood = lhood_
                break
            else:
                lhood = lhood_
        print "Time taken: ", (time.time() - start)

        return lhood, Z, O_

