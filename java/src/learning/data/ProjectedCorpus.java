/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.RandomFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Random;
import java.util.Vector;
import java.util.Date;

import java.io.Serializable;

import org.ejml.simple.SimpleMatrix;

/**
 * Stores a corpus in an integer array
 */
public class ProjectedCorpus extends Corpus implements Serializable {
  public int projectionDim;
  protected long[] seeds;

  protected ProjectedCorpus() {
    super();
  }

  public ProjectedCorpus( String[] dict, int[][] C, int d, long[] seeds ) {
    super( dict, C );
    projectionDim = d;

    // Stuff for lazy featurisation
    this.seeds = seeds;
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
  /**
   * If no seed was provided, use the RandomFactory rng.
   */
  public static ProjectedCorpus fromCorpus( Corpus C, int d ) {
    // Generate a set of seeds for each word
    long[] seeds = new long[ C.dict.length ];
    for(int i = 0; i < C.dict.length; i++ ) 
      seeds[i] = RandomFactory.rand.nextLong();

    return new ProjectedCorpus( C.dict, C.C, d, seeds );
  }

  /**
   * Get the d-dimensional feature vectore for the i-th index word
   */
  public double[] featurize( int i ) {
    Random rnd = new Random( seeds[i] );
    double[] x = new double[projectionDim];
    for(int j = 0; j < projectionDim; j++ )
      x[j] = rnd.nextGaussian();
    //x[j] = 10 * rnd.nextDouble();
    // Normalize x
    MatrixOps.makeUnitVector( x );
    return x;
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
      double[] feature = featurize(i);
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

