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
import fig.basic.LogInfo;

import org.ejml.simple.SimpleMatrix;

/**
 * Stores a corpus in an integer array
 */
public class ProjectedCorpus extends Corpus implements Serializable, RealSequenceData {
  private static final long serialVersionUID = 2L;
  public int projectionDim;
  protected long masterSeed;

  protected Featurizer featurizer;

  public double[][] P; // stored as a matrix for efficiency reasons
  public SimpleMatrix Pinv; // Projection Matrix

  protected ProjectedCorpus() {
    super();
  }

  public ProjectedCorpus(ProjectedCorpus PC) {
    super((Corpus)PC);
    this.projectionDim = PC.projectionDim;
    this.masterSeed = PC.masterSeed;
    this.P = PC.P;
    this.Pinv = PC.Pinv;
  }

  public ProjectedCorpus(Corpus C, int d, long seed) {
    super(C);
    this.projectionDim = d;
    this.masterSeed = seed;

    // y = P^T x
    this.P = makeProjection(super.getDimension());
    // x = (P^T)^+ y
    this.Pinv = (new SimpleMatrix(P)).transpose().pseudoInverse();
  }
  public ProjectedCorpus(Corpus C, int d, long seed, Featurizer featurizer) {
    super(C);
    this.projectionDim = d;
    this.masterSeed = seed;
    this.featurizer = featurizer;

    // y = P^T x
    this.P = makeProjection(super.getDimension() + featurizer.numFeatures());
    // x = (P^T)^+ y
    this.Pinv = (new SimpleMatrix(P)).transpose().pseudoInverse();
  }

  /**
   * Construct a random (Gaussian) projection matrix.
   */ 
  protected double[][] makeProjection(int N) {
    int D = projectionDim;
    // constructing here (and not RandomFactory) because we want to use 
    // a RNG with our specific seed.
    Random rnd = new Random( masterSeed );
    double[][] P_ = new double[N][D];
    for(int n = 0; n < N; n++ ) {
      for(int d = 0; d < D; d++ ) {
        P_[n][d] = rnd.nextGaussian();
      }
      MatrixOps.makeUnitVector(P_[n]);
    }

    return P_;
  }

  /**
   * Get the d-dimensional feature vectore for the i-th index word
   */
  public double[] featurize( int i ) {
    double[] x = P[i];

    // Find all the features for i.
    if( featurizer != null ) {
      int baseDim = super.getDimension();
      for( Integer f : featurizer.features( dict[i] ) ){
        // Include feature from i.
        MatrixOps.add( x, P[baseDim + f] );
      }
    }
    return x;
  }

  /**
   * Get the d-dimensional feature vectore for the i-th index word
   */
  public SimpleMatrix unfeaturize( SimpleMatrix v ) {
    return Pinv.mult(v);
  }
  public SimpleMatrix unfeaturize( double[] v ) {
    return unfeaturize( MatrixFactory.fromVector( v ).transpose() );
  }

  public int getDimension() {
    return projectionDim;
  }
  public int getInstanceCount() {
    return super.getInstanceCount();
  }
  public int getInstanceLength(int instance) {
    return C[instance].length;
  }
  public double[] getDatum(int instance, int index) {
    return featurize(C[instance][index]);
  }

}

