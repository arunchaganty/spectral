"""
The Brown Clustering algorithm (Brown et. al, 1992) with ideas from
Liang 2005.
"""

import scipy as sc
from scipy import array
import scipy.sparse
from scipy.sparse import lil_matrix
import ipdb
import util
from operator import itemgetter

class BrownClusteringAlgorithm:
    """Maintain as a class to keep track of all state variables"""

    def __init__( self, C ):
        """Initialise the algorithm for the corpus C and vocabulary V""" 
        self.C = C
        self.V = max( map( lambda d: d.max(), C ) ) + 1

        self.u = self.compute_unigrams( )

    def compute_unigrams( self ):
        """Get the unigram counts"""
        u = sc.zeros( self.V )
        for d in self.C:
            u[ d ] += 1
        # Normalise
        self.N = u.sum()
        u /= self.N
        return u

    def compute_bigrams( self, clusters ):
        """Get the bigram between clusters"""
        nC = len(clusters)
        b = sc.zeros( (nC, nC) )

        # Create an inverse lookup table
        clusterOf = sc.zeros(self.V)
        for i in xrange(len(clusters)):
            clusterOf[list(clusters[i])] = i

        # TODO: Optimise
        for d in self.C:
            for i in xrange( len(d) -1 ):
                b[ clusterOf[d[i]], clusterOf[d[i+1]] ] += 1
        b /= self.N**2
        return b

    def merge_clusters( self, u, b, L, c, c_ ):
        """Compute the weight of the merge c \cup c'
        P(c \cup c') = P(c) + P(c')
        P(c \cup c', c'') = P(c, c'') + P(c', c'')
        P(c \cup c', c \cup c') = P(c, c) + P(c, c') + P(c', c) + P(c', c') 

        Returns the adjusted u, b and L (with the union cluster pushed to the last column)
        """
        n, = u.shape
        w = sc.zeros( n - 1 )

        # The new cluster selection set
        C_ = range( n )
        C_.remove( c )
        C_.remove( c_ )

        # Update u
        u_ = sc.zeros( n-1 )
        u_[:n-1] = u[C_] 
        u_[n-1] = u[c] + u[c_]

        # Update b
        b_ = sc.zeros( (n-1,n-1) )
        b_[:n-1, :n-1] = b[C_, C_]
        b_[n-1, :n-1] = b[c, C_] + b[c_, C_]
        b_[:n-1, n-1] = b[C_, c] + b[C_, c_]
        b_[n-1, n-1] = b[c, c] + b[c_, c_] + b[c_, c] + b[c, c_]

        def w( x, y ):
            z =  b[x, y] * ( log( b[x, y] ) - log( u[x] ) - log( u[y] ) )
            z += b[y, x] * ( log( b[y, x] ) - log( u[x] ) - log( u[y] ) )
            return z
        def w_( x, y ):
            z = b_[x, y] * ( log( b_[x, y] ) - log( u_[x] ) - log( u_[y] ) )
            z += b_[y, x] * ( log( b_[y, x] ) - log( u_[x] ) - log( u_[y] ) )
            return z
        def wm( x, x_, y ):
            z = (b[x, y] + b[x_, y]) * ( log( b[x, y] + b[x_, y] ) - log( u[x] + u[x_] ) - log( u[y] ) )
            z += (b[y, x] + b[y, x_]) * ( log( b[y, x] + b[y, x_] ) - log( u[x] + u[x_] ) - log( u[y] ) )
            return z
        def wm_( x, x_, y ):
            z = (b_[x, y] + b_[x_, y]) * ( log( b_[x, y] + b_[x_, y] ) - log( u_[x] + u_[x_] ) - log( u_[y] ) )
            z += (b_[y, x] + b_[y, x_]) * ( log( b_[y, x] + b_[y, x_] ) - log( u_[x] + u_[x_] ) - log( u_[y] ) )
            return z

        # Update L
        L[:n-1,:n-1] = L[C_, C_]
        for d in xrange( n-2 ):
            for d_ in xrange( c+1, n-2 ):
                # e, e_ are the indices of d, d_ in the original array
                e, e_ = C_.index(d), C_.index( d_ )
                # \delta L = w(d \cup d', c \cup c') - w(d \cup d', c) - w(d \cup d', c') 
                #   - (w(d , c \cup c') - w(d , c) - w(d , c') )
                #   - (w(d', c \cup c') - w(d', c) - w(d', c') )
                L[d,d_] += wm_(d, d_, -1) - wm(e, e_, c) -  wm(e, e_, c_) 
                L[d,d_] += - (w_(d, -1) - w(e, c) -  w(e, c_))
                L[d,d_] += - (w_(d_, -1) - w(e_, c) -  w(e_, c_))
        for c in xrange( n-2 ):
            c_ = -1
            # The new cluster selection set
            C_ = [i for i in range(n-1) if i != c and i != c_ ]
            L[c, c_] = w_(c, c_, C_).sum() + w_(c, c_, [c,c_] ).sum() / 2

        return u_, b_, L

    def compute_merge_cost( self, u, b ):
        """Compute the weight of the merge c \cup c' for each pair c, c_
        P(c \cup c') = P(c) + P(c')
        P(c \cup c', c'') = P(c, c'') + P(c', c'')
        P(c \cup c', c \cup c') = P(c, c) + P(c, c') + P(c', c) + P(c', c') 

        L(c,c') = \sum_{d \in C'} w( c \cup c', d ) -  \sum_{d \in C} ( w( c, d ) + w( c', d ) )

        Returns the adjusted weights, u and b (with the union cluster pushed to the last column)
        """
        n = u.shape
        L = sc.inf * sc.ones( (n,n) )

        def w( x, x_, y ):
            """Compute the mutual information weights of merger,
            w(c,c') = P(c,c') log p(c,c')/p(c)p(c') + P(c',c) log p(c',c)/p(c)p(c') if c != c'
            w(c,c') = P(c,c) log p(c,c)/p(c)p(c) if c == c'
            """
            z = (b[x, y] + b[x_, y]) * ( log( b[x, y] + b[x_, y] ) - log( u[x] + u[x_] ) - log( u[y] ) )
            z += (b[y, x] + b[y, x_]) * ( log( b[y, x] + b[y, x_] ) - log( u[x] + u[x_] ) - log( u[y] ) )
            return z

        for c in xrange( n ):
            for c_ in xrange( c+1, n ):
                # The new cluster selection set
                C_ = [i for i in range(n) if i != c and i != c_ ]
                L[c, c_] = w(c, c_, C_).sum() + w(c, c_, [c,c_] ).sum() / 2
        return L

    def run( self, k, W ):
        """Run the algorithm until 'k' clusters Initialising the top 'W'
        words as clusters"""

        # Initialise the top W as individual clusters 
        clusters = range(self.V)
        clusters.sort( key = self.u.__getitem__ )
        clusters = map( lambda x: set([x]), clusters[:W] )
        # Put the rest in a # separate cluster
        remainder = set(range(self.V)).difference( reduce( lambda x, y: x.union(y), clusters ) )
        clusters.append( remainder )

        u = array( map( lambda x: self.u[list(x)].sum(), clusters ) )
        b = self.compute_bigrams( clusters )

        # L is memoised
        N = W+1
        L = self.compute_merge_cost( u, b )
        # Iterate until k clusters
        for i in xrange( N, k-1, -1 ):
            # Choose the cluster pair with the smallest cost
            c = L.argmin()
            c, c_ = c / N, c % N

            merged = clusters[c].union( clusters[c_] )
            clusters.remove(c)
            clusters.remove(c_)
            clusters.append( merged )

            # Memoised reduction step
            u, b, L = merge_clusters( u, b, L, c, c_ )

        return clusters

def test_unigram_counts():
    C = array([ 
            array([ 0, 1, 2, 3, 4 ]),
            array([ 0, 2, 3 ]),
            array([ 0, 3, 4 ]),
            array([ 2, 3, 4 ]) 
            ])
    bc = BrownClusteringAlgorithm( C )

    u = array( [3, 1, 3, 4, 3] )
    u = u/u.sum()

    assert( sc.allclose( u, bc.u ) )

def test_get_bigram_counts():
    C = array([ 
            array([ 0, 1, 2, 3, 4 ]),
            array([ 0, 4, 2 ]),
            array([ 4, 0, 3 ]),
            array([ 3, 2, 4 ]) 
            ])
    V = 5
    bc = BrownClusteringAlgorithm( C )

    b = array( [[ 0.,  0.,  1.,  1.],
       [ 1.,  0.,  0.,  2.],
       [ 1.,  0.,  0.,  1.],
       [ 0.,  1.,  2.,  0.]])
    N = sum( map(len, C) )
    b /= N
    clusters = [ set([3]), set([0]), set([2]), set([1,4]) ]
    b_ = bc.compute_bigrams( clusters )
    assert( sc.allclose( b, b_ ) )
    
def test_brown_clustering():
    fname = "test-data/text-1e4.npz"
    F = sc.load( fname )
    C, D = F['C'], F['D']
    k = 100
    W = 1000

    bc = BrownClusteringAlgorithm( C )
    bc.run( k, W )

# Parsing and data prepping functions

def parse_text( fname, n ):
    """Parse a text file containing a sentence on each line. 
    Output the file as a list of arrays with integer word indices and
    the corresponding dictionary."""

    dictionary = {}
    inv_dictionary = []
    def get_idx( w ):
        if not w in dictionary:
            dictionary[w] = len(dictionary)
            inv_dictionary.append( w )
        return dictionary[w]

    corpus = []

    f = open( fname )
    for line in f.xreadlines():
        # Read at most n documents
        if len(corpus) > n:
            break
        words = line.split()
        words = array( map( get_idx, words ) )
        if len(words) > 0:
            corpus.append( words )
    f.close()

    return array(corpus), array(inv_dictionary)

def test_parse_text():
    """Read a test text file and attempt to parse it"""
    fname = "test-data/text-1e6.txt"
    n = 1e6
    corpus, dictionary = parse_text( fname, n )
    assert( len(corpus) < n )
    assert( max( map( max, corpus ) ) == len(dictionary) - 1)

def save_corpus(fname, ofname, n):
    """Read @n documents of corpus in @fname and save the parsed arrays in @ofname"""
    corpus, dictionary = parse_text( fname, n )
    sc.savez( ofname, C = corpus, D = dictionary )

if __name__ == "__main__":
    save_corpus( "test-data/text-1e6.txt", "test-data/text-1e4.npz", 1e4 )
    save_corpus( "test-data/text-1e6.txt", "test-data/text-1e6.npz", 1e6 )
#    import argparse, time
#    parser = argparse.ArgumentParser()
#    parser.add_argument( "fname", help="Input file (as npz)" )
#    parser.add_argument( "ofname", help="Output file (as npz)" )
#    parser.add_argument( "--samples", type=float, help="Limit number of samples" )
#    parser.add_argument( "--subsamples", default=-1, type=float, help="Subset of samples to be used" )
#
#    args = parser.parse_args()
#
#    logger = DataLogger(args.ofname)
#    main( args.fname, int(args.samples), int(args.subsamples), args )
#
