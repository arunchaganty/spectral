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
import fig.basic.OptionsParser;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.IOException;
import java.io.Serializable;

import java.util.Random;
import java.util.Arrays;

/**
 * A hidden markov model.
 */
public class HiddenMarkovModel {
		
  // Model parameters
	public static class Params implements Serializable {
		private static final long serialVersionUID = 2704243416017266665L;

    public double[] pi;
    public double[][] T;
    public double[][] O;
    public int[][] map = null;
    
    public int stateCount;
    public int emissionCount;

		Params(int stateCount, int emissionCount) {
			this.stateCount = stateCount;
			this.emissionCount = emissionCount;
			pi = new double[stateCount];
			T = new double[stateCount][stateCount];
			O = new double[stateCount][emissionCount];
		}
		
		@Override
		public Params clone() {
			Params p = new Params(stateCount, emissionCount);
			p.pi = pi.clone();
			p.T = T.clone();
			p.O = O.clone();
			
			return p;
		}
		
		public static Params uniformWithNoise(int stateCount, int emissionCount, double noise) {
			Params p = new Params(stateCount, emissionCount);
			
			// Initialize each as uniform plus noise 
			// Pi
			for(int i = 0; i < stateCount; i++) {
				p.pi[i] = 1.0/stateCount;
				// Dividing the noise by the sqrt(size) so that the total noise is as given
				p.pi[i] += RandomFactory.randn( noise / Math.sqrt(stateCount) );
				// Ensure positive
				p.pi[i] = Math.abs( p.pi[i] );
			}
			MatrixOps.normalize( p.pi );
			
			// T
			for(int i = 0; i < stateCount; i++) {
				for(int j = 0; j < stateCount; j++) {
					p.T[i][j] = 1.0/stateCount;
					// Dividing the noise by the sqrt(size) so that the total noise is as given
					p.T[i][j] += RandomFactory.randn( noise / Math.sqrt(stateCount) );
					// Ensure positive
					p.T[i][j] = Math.abs( p.T[i][j] );
				}
				MatrixOps.normalize( p.T[i] );
			}
			
			// O
			for(int i = 0; i < stateCount; i++) {
				for(int j = 0; j < emissionCount; j++) {
					p.O[i][j] = 1.0/emissionCount;
					// Dividing the noise by the sqrt(size) so that the total noise is as given
					p.O[i][j] += RandomFactory.randn( noise / Math.sqrt(emissionCount) );
					// Ensure positive
					p.O[i][j] = Math.abs( p.O[i][j] );
				}
				MatrixOps.normalize( p.O[i] );
			}
			return p;
		}
		
		public static Params fromCounts(int stateCount, int emissionCount, int[][] X, int[][] Z, boolean shouldSmooth ) {
			// Normalize each pi, X and Z
			int N = X.length;
			
			Params p = new Params(stateCount, emissionCount);
			
			// For each sequence in X and Z, increment those counts
			for( int i = 0; i < N; i++ ) {
				for( int j = 0; j < Z[i].length; j++ )
				{
					p.pi[Z[i][j]] += 1;
					if( j < Z[i].length - 1)
					{
						p.T[Z[i][j]][Z[i][j+1]] += 1;
					}
					p.O[Z[i][j]][X[i][j]] += 1;
				}
			}
			
			// Add a unit count to everything
			if( shouldSmooth ) {
				for(int i = 0; i < stateCount; i++ ) {
					p.pi[i] += 1;
					for( int j = 0; j < stateCount; j++ ) p.T[i][j] += 1;
					for( int j = 0; j < emissionCount; j++ ) p.O[i][j] += 1;
				}
			}
			
			// normalize
			MatrixOps.normalize( p.pi );
			for( int i = 0; i < stateCount; i++ ) {
				MatrixOps.normalize( p.O[i] );
				MatrixOps.normalize( p.T[i] );
			}
			
			return p;
		}
	}
  public static class GenerationOptions {
    @Option( gloss = "Number of hidden states" )
    int stateCount = 2;
    @Option( gloss = "Dimension of observed variables" )
    int emissionCount = 3;

    @Option( gloss = "Noise parameter" )
    double noise = 1.0;

    public GenerationOptions(int stateCount, int emissionCount) {
      this.stateCount = stateCount;
      this.emissionCount = emissionCount;
    }

    public GenerationOptions(int stateCount, int emissionCount, double noise) {
      this.stateCount = stateCount;
      this.emissionCount = emissionCount;
      this.noise = noise;
    }
  }


  Random rnd = new Random();

  protected final Params params;

  public HiddenMarkovModel( Params p ) {
    this.params = p;
  }
  public HiddenMarkovModel( int stateCount, int emissionCount ) {
		params = new Params( stateCount, emissionCount );
  }

  public static HiddenMarkovModel generate( GenerationOptions options ) {
		Params params = Params.uniformWithNoise( options.stateCount, options.emissionCount, options.noise );
    return new HiddenMarkovModel( params );
  }

  public Params getParams() {
    return params;
  }
  public int getStateCount() {
    return params.stateCount;
  }
  public int getEmissionCount() {
    return params.emissionCount;
  }
  public SimpleMatrix getPi() {
    return MatrixFactory.fromVector( params.pi ).transpose();
  }
  public SimpleMatrix getT() {
    return (new SimpleMatrix( params.T )).transpose();
  }
  public SimpleMatrix getO() {
    return (new SimpleMatrix( params.O )).transpose();
  }

	/**
	 * Generate a single observation sequence of length n
	 * @param n
	 * @return
	 */
	public int[] sample(int n) {
		int[] output = new int[n];
		
		// Pick a start state
		int state = RandomFactory.multinomial(params.pi);
		
		for( int i = 0; i < n; i++)
		{
			// Generate a word
			int o = RandomFactory.multinomial( params.O[state] );
			if( params.map != null )
				output[i] = params.map[state][o];
			else
				output[i] = o;
			// Transit to a new state
			state = RandomFactory.multinomial( params.T[state] );
		}
		
		return output;
	}

  /**
   * Sample both the observed and hidden variables.
   */
	public Pair<int[], int[]> sampleWithHiddenVariables(int n) {
		int[] observed = new int[n];
		int[] hidden = new int[n];
		
		// Pick a start state
		int state = RandomFactory.multinomial(params.pi);
		
		for( int i = 0; i < n; i++)
		{
			// Generate a word
			int o = RandomFactory.multinomial( params.O[state] );
      hidden[i] = state;

			if( params.map != null )
				o = params.map[state][o];
      observed[i] = o;

			// Transit to a new state
			state = RandomFactory.multinomial( params.T[state] );
		}

		return new Pair<>( observed, hidden );
	}
	
	/**
	 * Use the Viterbi dynamic programming algorithm to find the hidden states for o.
	 * @param o
	 * @return
	 */
	public int[] viterbi( final int[] o ) {
		// Store the dynamic programming array and back pointers
		double [][] V = new double[o.length][params.stateCount];
		int [][] Ptr = new int[o.length][params.stateCount];
		
		// Initialize with 0 and path length
		for( int s = 0; s < params.stateCount; s++ )
		{
			// P( o_0 | s_k ) \pi(s_k)
			V[0][s] = params.O[s][o[0]] * params.pi[s];
			Ptr[0][s] = -1; // Doesn't need to be defined.
		}
		
		// The dynamic program to find the optimal path
		for( int i = 1; i < o.length; i++ ) {
			for( int s = 0; s < params.stateCount; s++ )
			{
				// Find the max of T(s | s') V_(i-1)(s')
				double T_max = 0.0;
				int S_max = -1;
				for( int s_ = 0; s_ < params.stateCount; s_++ ) {
					if( params.T[s_][s] * V[i-1][s_] > T_max ) {
						T_max = params.T[s_][s] * V[i-1][s_];
						S_max = s_;
					}
				}
				
				// P( o_i | s_k ) = P(o_i | s) *  max_j T(s | s') V_(i-1)(s')
				V[i][s] = params.O[s][o[i]] * T_max;
				Ptr[i][s] = S_max; 
			}
		}
		
		int[] z = new int[o.length];
		// Choose the best last state and back track from there
		z[o.length-1] = MatrixOps.argmax(V[o.length-1]);
		for(int i = o.length-1; i >= 1; i-- ) 
			z[i-1] = Ptr[i][z[i]];
		
		return z;
	}
	
	/**
	 * Use the forward-backward algorithm to find the posterior probability over states
	 * @param o
	 * @return
	 */
	public double[][] forwardBackward( final int[] o ) {
		// Store the forward probabilities
		double [][] f = new double[o.length][params.stateCount];
		// Normalization constants
		double [] c = new double[o.length];
		
		// Initialise with the initial probabilty
		for( int s = 0; s < params.stateCount; s++ ) {
			f[0][s] = params.pi[s] * params.O[s][o[0]];
		}
		for( int s = 0; s < params.stateCount; s++ ) c[0] += f[0][s];
    for( int s = 0; s < params.stateCount; s++ ) {
      f[0][s] /= c[0];
    }
		
		// Compute the forward values as f_t(s) = sum_{s_} f_{t-1}(s_) * T( s | s_ ) * O( y | s )
		for( int i = 1; i < o.length; i++ ) {
      //System.out.printf( "f[%d]\n", i-1 );
      //MatrixOps.printVector( f[i-1] );
			for( int s = 0; s < params.stateCount; s++ ) {
				f[i][s] = 0.0;
				for( int s_ = 0; s_ < params.stateCount; s_++ ) {
					f[i][s] += f[i-1][s_] * params.T[s_][s];
				}
				f[i][s] *= params.O[s][o[i]];
			}
			// Compute normalisation constant
			for( int s = 0; s < params.stateCount; s++ ) c[i] += f[i][s];
			// Normalise
      for( int s = 0; s < params.stateCount; s++ ) {
        f[i][s] /= c[i];
      }
    }
		
		double [][] b = new double[o.length][params.stateCount];
		for( int s = 0; s < params.stateCount; s++ ) {
			b[o.length-1][s] = 1.0;
		}
		for( int i = o.length-2; i >= 0; i-- ) {
      //System.out.printf( "b[%d]\n", i+1 );
      //MatrixOps.printVector( b[i+1] );
			for( int s = 0; s < params.stateCount; s++ ) {
				for( int s_ = 0; s_ < params.stateCount; s_++ ) {
					b[i][s] += b[i+1][s_] * params.T[s][s_] * params.O[s_][o[i+1]];
				}
			}
			// Normalise
      for( int s = 0; s < params.stateCount; s++ ) 
        b[i][s] /= c[i+1];
		}

		double[][] z = new double[o.length][params.stateCount];
		for(int i = 0; i < o.length; i++ ) {
			for(int s = 0; s < params.stateCount; s++ )
				z[i][s] = f[i][s] * b[i][s];
      //System.out.printf( "z[%d]\n", i );
      //MatrixOps.printVector( z[i] );
    }
		return z;
	}

  /** 
   * Print the likelihood of a sequence.
   */
	public double likelihood( final int[] o, final int[] z ) {
    assert( o.length == z.length );
    double lhood = 0.0;
		for(int i = 0; i < o.length; i++ ) {

      if( i == 0 )
        lhood += Math.log( params.pi[z[i]] );
      else
        lhood += Math.log( params.T[z[i-1]][z[i]] );
      lhood += Math.log( params.O[z[i]][o[i]] );
    }
		return lhood;
  }
	
	public static HiddenMarkovModel learnFullyObserved( int stateCount, int emissionCount, int[][] X, int[][] Z, 
			boolean shouldSmooth) {
		Params p = Params.fromCounts( stateCount, emissionCount, X, Z, shouldSmooth);
		
		return new HiddenMarkovModel(p);
	}	
		
	public static HiddenMarkovModel learnFullyObserved( int stateCount, int emissionCount, int[][] X, int[][] Z, 
			boolean shouldSmooth, int compressedEmissionCount) {
		
		Params p = Params.fromCounts( stateCount, emissionCount, X, Z, shouldSmooth);
		
		// If compressing, then sort the emissions for each state and keep only the top compressedEmissionCount
		double[][] O_ = new double[stateCount][compressedEmissionCount];
		// Sparse map for compressed emissions
		int[][] map = new int[stateCount][compressedEmissionCount];
		
		for( int i = 0; i < stateCount; i++ ) {
			Integer[] words_ = new Integer[emissionCount];
			for(int j = 0; j <emissionCount; j++) words_[j] = j;
			
			// Choose top k words
			Arrays.sort(words_, new Misc.IndexedComparator(p.O[i]) );
			for( int j = 0; j < compressedEmissionCount; j++ ) {
				O_[i][j] = p.O[i][words_[j]];
				map[i][j] = words_[j];
			}
			MatrixOps.normalize( O_[i] );
		}
		
		Params p_ = new Params(stateCount, compressedEmissionCount);
		p_.pi = p.pi; p_.T = p.T; p_.O = O_; p_.map = map;
		
		return new HiddenMarkovModel(p_);
	}

  public static class OutputOptions {
  }
  /**
   * Generates data with given specifications to stdout. 
   */
  public static void main( String[] args ) {
  }
}

