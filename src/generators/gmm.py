"""
Generate data from a Gaussian mixture model
"""

import ipdb
import scipy as sc
from scipy import matrix, array
from scipy.linalg import norm 

from IPython.core.debugger import Tracer
debug = Tracer()

multinomial = sc.random.multinomial
multivariate_normal = sc.random.multivariate_normal 
dirichlet = sc.random.dirichlet

import spectral.linalg as sl
import spectral.random as sr

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

class GaussianMixtureModel( MixtureModel ):
    """Generic mixture model with N components"""
    def __init__( self, weights, means, sigmas ):
        """Create a mixture model for components using given weights"""
        MixtureModel.__init__(self, weights, means)
        self.sigmas = sigmas

    def sample( self, n ):
        """Sample n samples from the mixture model"""
        cnts = multinomial( n, self.weights )

        # Shuffle all of the data 
        # Row permutation only
        shuffle = sr.permutation( n ), None

        # Data for each view
        data = [ ]
        for view in xrange(self.n_views):
            data_ = []
            for comp in xrange( len( self.weights ) ):
                mean, std = self.means[ view ].T[ comp ], self.sigmas[ view ]
                data_.append( multivariate_normal( mean, std, cnts[ comp ] ) )
            data_ = matrix( sc.vstack( data_ ) )
            data_ = sl.apply_matrix_permutation( shuffle, data_ )
            data.append( data_ )
        return data

    @staticmethod
    def generate( k, d, views = 3 ):
        """Generate a mixture of k gaussians""" 

        means, sigmas = [], []
        for i in xrange( views ):
            means.append( sc.randn( d, k ) )
            sigmas.append( sc.eye( d ) ) # 0.1 * mr.wishart( d, sc.eye(d) )
        weights = dirichlet( sc.ones(k) * 10 ) # Scale factor of 10

        return GaussianMixtureModel( weights, means, sigmas )

def recovery_error( x, y ):
    """Return the difference between canonical forms of x and y"""
    x = array( x )
    y = array( y )
    x = sl.canonicalise( x )
    y = sl.canonicalise( y )

    return norm( x - y )


class TopicModel( MixtureModel ):
    """A simple topic model where each topic is represented as a
    multinomial and topics are independent and drawn according to some
    multinomial distribution."""

    def __init__( self, weights, topics ):
        """Create a mixture model for components using given weights"""

        # Number of topics and dictionary size
        self.W, self.K = topics.shape
        assert( self.W > self.K )

        self.topics = topics
        MixtureModel.__init__(self, weights, topics)

    def sample( self, n, words = 100 ):
        """Sample n samples from the topic model with the following process,
        (a) For each document, draw a particular topic
        (b) Draw w words at random from the topic
        """
        # Draw the number of documents to be drawn for each topic
        cnts = multinomial( n, self.weights )

        docs = []
        for (k, cnt) in zip( xrange(self.K), cnts ):
            # For each document of type k
            for i in xrange( cnt ):
                # Generate a document with `words` words from the
                # topic
                docs.append( multinomial( words, self.topics.T[k] ) )

        # Shuffle all of the data (not really necessary)
        docs = sc.column_stack(docs)
        sc.random.shuffle(docs)

        return docs

    @staticmethod
    def generate( k, n, scale = 10, prior = "uniform" ):
        """Generate k topics, each with n words""" 

        if prior == "uniform":
            # Each topic is a multinomial generated from a Dirichlet
            topics = sc.column_stack( [ dirichlet( sc.ones( n ) * scale
                ) for i in xrange( k ) ] )
            # We also draw the weights of each topic
            weights = dirichlet( sc.ones(k) * scale ) 

            return TopicModel( weights, topics )
        else:
            # TODO: Support the anchor word assumption.
            raise NotImplementedError

def test_topic_generator_dimensions( ):
    """Test the TopicModel generator"""

    tm = TopicModel.generate( 10, 1000 )
    assert( tm.topics.shape == (1000, 10) )
    assert( tm.weights.shape == (10,) )

    docs = tm.sample( 100 )
    # Each document is a column
    assert( docs.shape == (1000, 100) )  
    # Each doc should have 100 words
    assert( sc.all(docs.sum(0) == 100) )

