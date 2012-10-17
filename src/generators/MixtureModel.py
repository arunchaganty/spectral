
import scipy as sc
from scipy import matrix, array
from scipy.linalg import norm 

class MixtureModel:
    """Generic mixture model that contains a bunch of weighted means"""
    def __init__( self, weights, means ):
        """Create a mixture model for components using given weights"""
        assert( len(weights) == len(means[0].T) )
        self.n_views = len(means)
        self.weights = array( weights )
        self.means = array( means )

    def sample( self, n ):
        """Sample n samples from the mixture model"""
        raise NotImplementedError

    @property
    def get_means( self ):
        """Get means"""
        return self.means

