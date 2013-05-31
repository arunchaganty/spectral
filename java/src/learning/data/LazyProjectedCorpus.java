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

import fig.basic.Option;

import org.ejml.simple.SimpleMatrix;

/**
 * Stores a corpus in an integer array
 */
public class LazyProjectedCorpus extends Corpus implements Serializable, RealSequenceData {
  private static final long serialVersionUID = 2L;
  public int projectionDim;
  protected long masterSeed;

  protected LazyProjectedCorpus() {
    super();
  }

  protected LazyProjectedCorpus(LazyProjectedCorpus PC) {
    super((Corpus)PC);
    this.projectionDim = PC.projectionDim;
    this.masterSeed = PC.masterSeed;
  }

  public LazyProjectedCorpus( String[] dict, int[][] C, int d, long seed ) {
    super( dict, C );
    projectionDim = d;

    // Stuff for lazy featurisation
    this.masterSeed = seed;
  }

  /**
   * Project a corpus onto a random set of d-dimensional vectors
   */
  public static LazyProjectedCorpus fromCorpus( Corpus C, int d, long seed ) {
    return new LazyProjectedCorpus( C.dict, C.C, d, seed );
  }

  /**
   * Get the d-dimensional feature vectore for the i-th index word
   */
  public double[] featurize( int i ) {
    Random rnd = new Random( masterSeed + i );
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

  public int getDimension() {
    return projectionDim;
  }
  public int getInstanceCount() {
    return C.length;
  }
  public int getInstanceLength(int instance) {
    return C[instance].length;
  }
  public double[] getDatum(int instance, int index) {
    return featurize(C[instance][index]);
  }

}

