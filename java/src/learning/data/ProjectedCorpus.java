/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Random;
import java.util.Vector;
import java.util.Date;

import org.ejml.simple.SimpleMatrix;

import com.spaceprogram.kittycache.KittyCache;

/**
 * Stores a corpus in an integer array
 */
public class ProjectedCorpus extends Corpus {

	public int projectionDim;
	protected Random rnd;
	protected long[] seeds;
	protected KittyCache<Integer, double[]> featureCache;

  public ProjectedCorpus( String[] dict, int[][] C, int d, long[] seeds ) {
    super( dict, C );
    projectionDim = d;

    // Stuff for lazy featurisation
		this.rnd = new Random();
		this.seeds = seeds;
		this.featureCache = new KittyCache<>( 1000 );
  }

  /**
   * Project a corpus onto a random set of d-dimensional vectors
   */
  public static ProjectedCorpus fromCorpus( Corpus C, int d, long seed ) {
    Random rnd = new Random( seed );

    // Generate a set of seeds for each word
		long[] seeds = new long[ C.dict.length ];
		for(int i = 0; i < C.dict.length; i++ ) 
			seeds[i] = rnd.nextLong();

    return new ProjectedCorpus( C.dict, C.C, d, seeds );
  }
  public static ProjectedCorpus fromCorpus( Corpus C, int d ) {
    return fromCorpus( C, d, (new Date()).getTime() );
  }

  /**
   * Get the d-dimensional feature vectore for the i-th index word
   */
  public double[] featurize( int i, boolean shouldCache ) {
    // Try getting it from the cache
		double[] x = featureCache.get(i);

		if( x == null ) {
      // Otherwise, set the seed and generate the random number
			rnd.setSeed(seeds[i]);
			x = new double[projectionDim];
			for(int j = 0; j < projectionDim; j++ )
				x[j] = rnd.nextGaussian();
				//x[j] = 10 * rnd.nextDouble();
      // Normalize x
      MatrixOps.makeUnitVector( x );
      
			if( shouldCache )
				featureCache.put(i, x, -1);
		}
		
		return x;
  }
  public double[] featurize( int i ) {
    return featurize( i, true );
  }

	/**
	 * Get the distribution over words for this feature
	 * @param feature
	 * @return
	 */
	public double[] getWordDistribution( double[] x ) {
		double[] z = new double[dict.length];

    MatrixOps.makeUnitVector( x );
		for(int i = 0; i < dict.length; i++ ) {
			double[] feature = featurize(i, false);
			z[i] = MatrixOps.dot(feature, x);
      if( z[i] < 0 ) z[i] = 0;
		}
    // Normalize into a probability distribution
    MatrixOps.projectOntoSimplex( z );
		return z;
	}
	public SimpleMatrix getWordDistribution( SimpleMatrix x ) {
		double[] x_ = x.getMatrix().data;
    return MatrixFactory.fromVector( getWordDistribution( x_ ) );
	}

}

