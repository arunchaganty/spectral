/**
 * learning.model
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import learning.Misc;
import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.RandomFactory;

import org.javatuples.*;
import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;

import fig.basic.Option;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.IOException;
import java.io.Serializable;

import java.util.Random;
import java.util.Arrays;

/**
 * A real-valued hidden markov model.
 */
public class RealHiddenMarkovModel extends HiddenMarkovModel {
  // Maps discrete feature indicies into a feature map
	public static class Features implements Serializable {
    private static final long serialVersionUID = 2L;

    int emissionCount;
    int dimension;
    public double[][] features; 
    public double noise = 0.0; 

    Features( int emissionCount, int dimension, double[][] features, double noise ) {
      this.emissionCount = emissionCount;
      this.dimension = dimension;
      this.features = features;
      this.noise = noise;
    }

    Features( int emissionCount, int dimension, SimpleMatrix features, double noise ) {
      this.emissionCount = emissionCount;
      this.dimension = dimension;
      this.features = MatrixFactory.toArray( features );
      this.noise = noise;
    }

		public static Features eye(int emissionCount, double noise) {
      return new Features( emissionCount, emissionCount, MatrixFactory.eye( emissionCount ), noise );
    }
		public static Features eye(int emissionCount) {
      return eye( emissionCount, 0.0 );
    }

		public static Features random(int emissionCount, int dimension, double noise) {
      return new Features( emissionCount, dimension, RandomFactory.rand( emissionCount, dimension ), noise );
    }
		public static Features random(int emissionCount, int dimension) {
      return random( emissionCount, dimension, 0.0 );
    }

		public static Features gaussian(int emissionCount, int dimension, double noise) {
      return new Features( emissionCount, dimension, RandomFactory.randn( emissionCount, dimension ), noise );
    }
		public static Features gaussian(int emissionCount, int dimension) {
      return random( emissionCount, dimension, 0.0 );
    }
  }
	public static class FeatureOptions {
    @Option( gloss = "Number of dimensions" )
    int dimension = -1;
    @Option( gloss = "Generation scheme" )
    String scheme = "eye";
    @Option( gloss = "Noise" )
    double noise = 0.0;

    public FeatureOptions() {}
    public FeatureOptions(int dimension, String scheme, double noise) {
      this.dimension = dimension;
      this.scheme = scheme;
      this.noise = noise;
    }
    public FeatureOptions(int dimension, String scheme) {
      this.dimension = dimension;
      this.scheme = scheme;
    }
  }

  protected final Features features;

  public RealHiddenMarkovModel( Params p, Features features ) {
    super(p);
    this.features = features;
  }
  public RealHiddenMarkovModel( int stateCount, int emissionCount, int dimension ) {
    super(new Params( stateCount, emissionCount ));
		features = Features.eye( emissionCount );
  }

  public int getDimension() {
    return features.dimension;
  }

  /**
   * Get the mean O when projected through the features matrix.
   */
  public SimpleMatrix getRealO() {
    double[][] M = new double[ params.stateCount ][ features.dimension ];
    for( int s = 0; s < params.stateCount; s++ ) {
      for( int o = 0; o < params.emissionCount; o++ ) {
        for( int d = 0; d < features.dimension; d++ ) {
          M[s][d] += params.O[s][o] * features.features[o][d];
        }
      }
    }

    return (new SimpleMatrix( M )).transpose();
  }

  public static RealHiddenMarkovModel generate( GenerationOptions options, FeatureOptions featureOptions ) {
		Params params = Params.uniformWithNoise( new Random(), options.stateCount, options.emissionCount, options.noise );
    Features features = null;

    if( featureOptions.scheme.equals( "eye" ) )
      features = Features.eye( options.emissionCount );
    else if( featureOptions.scheme.equals( "random" ) )
      features = Features.random( options.emissionCount, featureOptions.dimension );
    else if( featureOptions.scheme.equals( "gaussian" ) )
      features = Features.gaussian( options.emissionCount, featureOptions.dimension );
    else {
      throw new NoSuchMethodError();
    }

    return new RealHiddenMarkovModel( params, features );
  }

	/**
	 * Generate a single observation sequence of length n
	 * @param n
	 * @return
	 */
	public double[][] sampleReal(int N) {
    int D = features.dimension;
		double[][] output = new double[N][D];

    SimpleMatrix O_ = getRealO();
		
		// Pick a start state
		int state = RandomFactory.multinomial(params.pi);
		
		for( int n = 0; n < N; n++)
		{
      output[n] = MatrixFactory.toVector( MatrixOps.col( O_, state ) );
			// Generate a word
			// int o = RandomFactory.multinomial( params.O[state] );
			// if( params.map != null )
			// 	o = params.map[state][o];
      // output[n] = features.features[ o ];
      if( features.noise > 0.0 )
        for( int d = 0; d < D; d++ ) 
          output[n][d] += RandomFactory.randn(features.noise);
			// Transit to a new state
			state = RandomFactory.multinomial( params.T[state] );
		}
		
		return output;
	}
	public double[][][] sampleReal(int N, int M) {
    double[][][] output = new double[N][M][features.dimension];
    for( int n = 0; n < N; n++ ) {
      output[n] = sampleReal(M);
    }

    return output;
  }

  /**
   * Sample both the observed and hidden variables.
   */
	public Pair<double[][],int[]> sampleRealWithHiddenVariables(int N) {
    int D = features.dimension;
		double[][] output = new double[N][D];
		int[] hidden = new int[N];
		
		// Pick a start state
		int state = RandomFactory.multinomial(params.pi);
		
		for( int n = 0; n < N; n++)
		{
			// Generate a word
			int o = RandomFactory.multinomial( params.O[state] );
      hidden[n] = state;
			if( params.map != null )
				o = params.map[state][o];
      output[n] = features.features[ o ];
      if( features.noise > 0.0 )
        for( int d = 0; d < D; d++ ) 
          output[n][d] += RandomFactory.randn(features.noise);
			// Transit to a new state
			state = RandomFactory.multinomial( params.T[state] );
		}

		return new Pair<>( output, hidden );
	}
	public Pair<double[][][], int[][]> sampleRealWithHiddenVariables(int N, int M) {
    double[][][] observed = new double[N][M][features.dimension];
    int[][] hidden = new int[N][M];
    for( int n = 0; n < N; n++ ) {
      Pair<double[][], int[]> output = sampleRealWithHiddenVariables( M );
      observed[n] = output.getValue0();
      hidden[n] = output.getValue1();
    }

    return new Pair<>( observed, hidden );
  }
	
  /** 
   * Print the likelihood of a sequence.
   */
	public double likelihood( final double[][] o, final int[] z ) {
    // TODO
    throw new NoSuchMethodError();
  }
	
  /**
   * Generates data with given specifications to stdout. 
   */
  public static void main( String[] args ) {
  }
}


