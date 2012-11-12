"""
Application of the spectral decomposition method from Anandkumar, Hsu,
Kakade, "A Method of Moments for Mixture Models and Hidden Markov
Models" (2012) to forming word clusters.
"""

import scipy as sc 
import scipy.spatial
import scipy.linalg
from scipy import diag, array, zeros
from scipy.linalg import norm, inv, det
cdist = scipy.spatial.distance.cdist

from spectral.MultiView import sample_moments, recover_M3

from util import DataLogger

import time

logger = DataLogger("log")

def project_data( C, d, d_ ):
    """Project the words in the corpus C onto a d_ dimensional subspace"""

    Z = sc.random.rand( d, d_ ) 
    # Normalise the rows
    Z = (Z.T / (Z * Z).sum( 1 )).T

    # We compute the three views as trigrams with a sliding window. So,
    # the total number of instances is equal to the number words in a
    # document - 2.
    N = reduce( lambda N, c: N + len(c) - 2, C, 0 )

    X1, X2, X3 = zeros( (N, d_), dtype = sc.float32 ), zeros( (N, d_), dtype = sc.float32 ), zeros( (N, d_), dtype = sc.float32 )

    # Sliding window
    i = 0
    for c in C:
        n = len( c ) - 2
        X1[i : i+n, :] = Z[c[0:n]]
        X2[i : i+n, :] = Z[c[1:n+1]]
        X3[i : i+n, :] = Z[c[2:n+2]]
        # Increment offset
        i += n

    return X1, X2, X3, Z

def main(fname, ofname, N, k, d, params ):
    """Open a text file, parse it, create 3 views of the data, run"""

    f = sc.load( fname )
    C, D = f["C"], f["D"]
    # Note: The sentinel value of -1 just works.
    C = C[:N]

    # Set seed for the algorithm
    sc.random.seed( int( params.seed ) )

    # For each word i, project onto a linear subspace 
    X1, X2, X3, Z = project_data( C, len(D), d )

    # Run the MultiView algorithm with M3 = X2 (because we want to
    # recover O

    P13, P12, P132 = sample_moments( X1, X3, X2 )

    start = time.time()
    M2 = recover_M3( k, P13, P12, P132 )
    stop = time.time()
    print "Time: %d", stop - start 

    # Print the clusters 
    # The columns of M2 are "cluster centers" - match the nearest words
    # to this cluster center
    Zi = cdist( Z, M2.T )
    # TODO: Sort by distance
    Zi = Zi.argmin(1)

    f = open( ofname, "w" )
    for i in xrange( k ):
        f.write( ", ".join( D[Zi == i] ) + "\n" )
    f.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "fname", help="Input file (as npz)" )
    parser.add_argument( "ofname", help="Output file (as npz)" )
    parser.add_argument( "--k", type=float, help="Number of clusters" )
    parser.add_argument( "--d", type=float, help="Number of latent dimensions" )
    parser.add_argument( "--samples", default=-1, type=float, help="Number of samples to be used" )
    parser.add_argument( "--seed", default=time.time(), type=long,
            help="Seed used for algorithm (separate from generation)" )

    args = parser.parse_args()

    main( args.fname, args.ofname, int(args.samples), int(args.k), int(args.d), args )


