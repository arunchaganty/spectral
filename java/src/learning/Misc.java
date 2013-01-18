/**
 * learning.Misc
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning;

import java.util.Comparator;

/**
 * Miscellaneous utilities
 */
public class Misc {

  /**
   * Lookup table for computing the base-2 log
   */
  static final int[] LogTable256 = {
     -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  };

  /**
   * Get the i-th bit of a number b
   */
  public static int bit( int v, int i ) {
    return (v >>> i) & 1;
  }

  /**
   * Get the log-base 2 of a number v
   */
  public static int log2( int v ) {
    int r;     // r will be lg(v)
    int t, tt; // temporaries

    tt = v >>> 16;
    t = tt >>> 8;
    if(tt != 0)
    {
      r = (t != 0) ? 24 + LogTable256[t] : 16 + LogTable256[tt];
    } else {
      r = (t != 0) ? 8 + LogTable256[t] : LogTable256[v];
    }

    return r;
  }

	/**
	 * Comparator to sort a list based on keys in another list 
	 */
	public static class IndexedComparator implements Comparator<Integer> {
		double[] keys;
		
		public IndexedComparator(double[] keys) {
			this.keys = keys;
		}
		
		@Override
		public int compare(Integer o1, Integer o2) {
			if( keys[o2] - keys[o1] < 0 )
				return -1;
			else if( keys[o2] - keys[o1] > 0 )
				return 1;
			else
				return 0;
		}
	}

  /** 
   * Returns N choose K
   */
  public static int binomial( int n, int k ) {
    if( k == 1) return n;
    else if( k == 0) return 1;
    else {
      return n * binomial( n - 1, k - 1 ) / k;
    }
  }

  public static interface TraversalFunction {
    /**
     * This function is run every time a new leaf node in a traversal is
     * reached.
     */
    public void run( int[] values );
  }

  static void traverseMultiCombination( int N, int K, int n, int k, int[] accum, TraversalFunction fn ) {
    // Check if we're done allocating $K$
    if( k == K )
      fn.run( accum );
    // Choose n to be one of $[k, K]$
    else {
      // Allocate the remainder in this case
      if( n == N-1 ) {
        accum[ n ] = K - k;
        traverseMultiCombination( N, K, n+1, K, accum, fn );
      }
      else {
        for( int k_ = K-k; k_ >= 0; k_-- ) {
          // Allocate k_ - k to n
          accum[n] = k_;
          traverseMultiCombination( N, K, n+1, k + k_, accum, fn );
        } 
      }
      accum[n] = 0;
    }
  }

  /** 
   * Traverse all multi-combinations of $N$ elements such that exactly
   * $K$ are chosen.
   */
  public static void traverseMultiCombination( int N, int K, TraversalFunction fn ) {
    int[] accum = new int[ N ];
    traverseMultiCombination( N, K, 0, 0, accum, fn );
  }

}
