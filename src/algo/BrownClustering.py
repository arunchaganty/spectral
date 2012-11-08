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

def get_unigram_counts( C, V ):
    """Get the unigram and bigram counts"""
    unigram = sc.zeros( V )
    for d in C:
        unigram[ d ] += 1
    return unigram

def test_get_unigram_counts():
    C = array([ 
            array([ 0, 1, 2, 3, 4 ]),
            array([ 0, 2, 3 ]),
            array([ 0, 3, 4 ]),
            array([ 2, 3, 4 ]) 
            ])
    u = get_unigram_counts( C, 5 )
    assert( sc.allclose( u, [3, 1, 3, 4, 3] ) )

def get_counts( C, V ):
    """Get the unigram and bigram counts"""
    unigram = sc.zeros( V )
    bigram = lil_matrix( (V, V) )
    for d in C:
        for i in xrange( len( d ) ):
            unigram[ d[i] ] += 1
            if i+1 < len(d):
                bigram[ d[i], d[i+1] ] += 1
    return unigram, bigram

def test_get_counts():
    C = array([ 
            array([ 0, 1, 2, 3, 4 ]),
            array([ 0, 2, 3 ]),
            array([ 0, 3, 4 ]),
            array([ 2, 3, 4 ]) 
            ])
    u, b = get_counts( C, 5 )
    b = b.todense()
    assert( sc.allclose( u, [3, 1, 3, 4, 3] ) )
    assert( sc.allclose( b[0], [0, 1, 1, 1, 0] ) )
    assert( sc.allclose( b[1], [0, 0, 1, 0, 0] ) )
    assert( sc.allclose( b[2], [0, 0, 0, 3, 0] ) )
    assert( sc.allclose( b[3], [0, 0, 0, 0, 3] ) )
    assert( sc.allclose( b[4], [0, 0, 0, 0, 0] ) )

def brown_clustering( C, V, W ):
    """
    Perform a brown clustering of a corpus C
    The rows of C are documents with word indices.
    V is the vocabulary size
    W is the initial number of parameters
    Returns a set of k clusters of words
    """
    unigram = get_unigram_counts( C, V )

    # Initialise the top W as individual clusters and put the rest in a
    # separate cluster
    clusters = range(V)
    clusters.sort( key = unigram.__getitem__ )
    clusters = map( lambda x: set([x]), clusters[:W] )
    remainder = set(range(V)).difference( reduce( lambda x, y: x.union(y), clusters ) )
    clusters.append( remainder )

    # Compute bigrams between the clusters

def test_brown_clustering():
    fname = "test-data/text-1e4.npz"
    F = sc.load( fname )
    C, D = F['C'], F['D']

    V = len(D)
    W = 1000

    brown_clustering( C, V, W )

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

def save_corpus(fname, ofname, n):
    corpus, dictionary = parse_text( fname, n )
    sc.savez( ofname, C = corpus, D = dictionary )

def test_parse_text():
    fname = "test-data/text-snippets-1m.txt"
    n = 1e6
    corpus, dictionary = parse_text( fname, n )

    assert( len(corpus) < n )
    assert( max( map( max, corpus ) ) == len(dictionary) - 1)

#def main():
#    pass

if __name__ == "__main__":
    #save_corpus( "test-data/text-snippets-1m.txt", "test-data/text-1e6.npz", 1e6 )
    save_corpus( "test-data/text-snippets-1m.txt", "test-data/text-1e4.npz", 1e4 )
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
