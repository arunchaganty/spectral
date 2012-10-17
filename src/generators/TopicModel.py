
import scipy as sc
from scipy import matrix, array
from generators.MixtureModel import MixtureModel

multinomial = sc.random.multinomial
multivariate_normal = sc.random.multivariate_normal 
dirichlet = sc.random.dirichlet

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

def test_topic_model_generator_dimensions( ):
    """Test the TopicModel generator"""
    N = 100
    D = 1000
    K = 10
    W = 100

    tm = TopicModel.generate( K, D )
    assert( tm.topics.shape == (D, K) )
    assert( tm.weights.shape == (K,) )

    docs = tm.sample( N, words = W )
    # Each document is a column
    assert( docs.shape == (D, N) )  
    # Each doc should have 100 words
    assert( sc.all(docs.sum(0) == W) )

class LDATopicModel( TopicModel ):
    """The LDA topic model where the words in a document can come from
    several topics"""

    def __init__( self, weights, topics ):
        """Create a mixture model for components using given weights"""

        # Number of topics and dictionary size
        self.W, self.K = topics.shape
        assert( self.W > self.K )

        self.topics = topics
        TopicModel.__init__(self, weights, topics)

    def sample( self, n, words = 100 ):
        """Sample n samples from the topic model with the following process,
        (a) For each word in a document, draw a particular topic
        (b) Draw the word from that topic
        """

        docs = []
        for i in xrange( n ):
            # Draw the number of words to be drawn for each topic
            cnts = multinomial( words, self.weights )

            doc = sc.zeros( self.W )
            for (k, cnt) in zip( xrange(self.K), cnts ):
                # Draw some words from topic k
                doc += multinomial( cnt, self.topics.T[k] )

            docs.append( doc )

        docs = sc.column_stack(docs)

        return docs

    @staticmethod
    def generate( k, n, scale = 10, prior = "uniform" ):
        """Generate k topics, each with n words""" 
        tm = TopicModel.generate( k, n, scale, prior )
        return LDATopicModel( tm.weights, tm.topics )


def test_lda_topic_model_generator_dimensions( ):
    """Test the LDATopicModel generator"""
    N = 100
    D = 1000
    K = 10
    W = 100

    tm = LDATopicModel.generate( K, D )
    assert( tm.topics.shape == (D, K) )
    assert( tm.weights.shape == (K,) )

    docs = tm.sample( N, words = W )
    # Each document is a column
    assert( docs.shape == (D, N) )  
    # Each doc should have 100 words
    assert( sc.all(docs.sum(0) == W) )

