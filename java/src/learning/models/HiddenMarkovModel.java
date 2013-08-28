/**
 * learning.model
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import fig.prob.Multinomial;
import learning.Misc;
import learning.data.HasExactMoments;
import learning.data.HasSampleMoments;
import learning.linalg.*;
import learning.em.EMOptimizable;

import org.javatuples.*;
import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;

import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.basic.LogInfo;
import fig.basic.Fmt;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.IOException;
import java.io.Serializable;

import java.util.Random;
import java.util.Arrays;

/**
 * A hidden markov model.
 */
public class HiddenMarkovModel implements EMOptimizable, Serializable, HasExactMoments, HasSampleMoments {
  /**
   * Compute exact moments from your parameters.
   * @return - exact moments
   */
  @Override
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> computeExactMoments() {
    SimpleMatrix pi = getPi();
    SimpleMatrix dpi = MatrixFactory.diag(pi);
    SimpleMatrix O = getO();
    SimpleMatrix T = getT();

    SimpleMatrix w = T.mult(pi);
    SimpleMatrix dw = MatrixFactory.diag(w);

    // M_1 = O \diag( \pi ) T^T \diag(T\pi)^{-1}
    SimpleMatrix M1 =
            O.mult(dpi).mult(T.transpose())
                    .mult(dw.invert());
    SimpleMatrix M2 = O;
    SimpleMatrix M3 = O.mult(T);

    return MixtureOfGaussians.computeExactMoments(w, M1, M2, M3);
  }

  @Override
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> computeSampleMoments(int N) {
    // Generate this much data and compute moments
    SimpleMatrix P13 = new SimpleMatrix(getEmissionCount(), getEmissionCount());
    SimpleMatrix P12 = new SimpleMatrix(getEmissionCount(), getEmissionCount());
    SimpleMatrix P32 = new SimpleMatrix(getEmissionCount(), getEmissionCount());
    FullTensor P123 = new FullTensor(getEmissionCount(), getEmissionCount(), getEmissionCount());

    for(int i = 0; i < N; i++) {
      int[] words = sample(3);
      // Update counts in P.
      P13.set(words[0], words[2], P13.get(words[0], words[2]) + 1);
      P12.set(words[0], words[1], P12.get(words[0], words[1]) + 1);
      P32.set(words[2], words[1], P32.get(words[2], words[1]) + 1);
      P123.X[words[0]][words[1]][words[2]] += 1;
    }

    // Normalize
    P13 = P13.scale(1.0/P13.elementSum());
    P12 = P12.scale(1.0/P12.elementSum());
    P32 = P32.scale(1.0/P32.elementSum());
    P123 = P123.scale(1.0/P123.elementSum());

    assert( MatrixOps.equal( P13.elementSum(), 1.0 ) );
    assert( MatrixOps.equal( P12.elementSum(), 1.0 ) );
    assert( MatrixOps.equal( P32.elementSum(), 1.0 ) );
    assert( MatrixOps.equal( P123.elementSum(), 1.0 ) );

    return Quartet.with(P13, P12, P32, P123);
  }
//  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> computeSampleMoments(int N) {
//    // Generate this much data and compute moments
//    SimpleMatrix P12 = new SimpleMatrix(getEmissionCount(), getEmissionCount());
//    SimpleMatrix P13 = new SimpleMatrix(getEmissionCount(), getEmissionCount());
//    SimpleMatrix P23 = new SimpleMatrix(getEmissionCount(), getEmissionCount());
//    FullTensor P132 = new FullTensor(getEmissionCount(), getEmissionCount(), getEmissionCount());
//
//    for(int i = 0; i < N; i++) {
//      int[] words = sample(3);
//      // Update counts in P.
//      P12.set(words[0], words[1], P12.get(words[0], words[1]) + 1);
//      P13.set(words[0], words[2], P13.get(words[0], words[2]) + 1);
//      P23.set(words[1], words[2], P23.get(words[1], words[2]) + 1);
//      P132.X[words[0]][words[2]][words[1]] += 1;
//    }
//
//    // Normalize
//    P13 = P13.scale(1.0/P13.elementSum());
//    P12 = P12.scale(1.0/P12.elementSum());
//    P23 = P23.scale(1.0/P23.elementSum());
//    P132 = P132.scale(1.0/P132.elementSum());
//
//    return Quartet.with(P12, P13, P23, P132);
//  }

  /**
   * Params stores the parameters for the HMM model.
   */
	public static class Params implements Serializable {
		private static final long serialVersionUID = 2704243416017266665L;

    public double[] pi;
    public double[][] T;
    public double[][] O;
    public int[][] map = null;
    
    public int stateCount;
    public int emissionCount;

		public Params(int stateCount, int emissionCount) {
			this.stateCount = stateCount;
			this.emissionCount = emissionCount;
			pi = new double[stateCount];
			T = new double[stateCount][stateCount];
			O = new double[stateCount][emissionCount];
		}

		public Params(double[] pi, double[][] T, double[][] O) {
      assert( T.length == T[0].length );
			this.stateCount = T.length;
      assert( pi.length == stateCount );
      assert( O.length == stateCount );
			this.emissionCount = O[0].length;
      // Check if the parameters are valid!
      assert( MatrixOps.equal( MatrixOps.sum( pi ),1.0 ) );
      for( int s = 0; s < stateCount; s++ ) {
        assert( MatrixOps.equal( MatrixOps.sum( T[s] ),1.0 ) );
        assert( MatrixOps.equal( MatrixOps.sum( O[s] ),1.0 ) );
      }

			this.pi = pi;
			this.T = T;
			this.O = O;
		}

    public boolean isValid() {
      boolean valid = true;
      double sum;
      sum = MatrixOps.sum(pi);
      valid &= MatrixOps.equal(sum, 1.0);
      assert valid;
      for( int s = 0; s < stateCount; s++ ) {
        sum = MatrixOps.sum(T[s]);
        valid &=  MatrixOps.equal(sum, 1.0);
        assert valid;
        sum = MatrixOps.sum(O[s]);
        valid &=  MatrixOps.equal(sum, 1.0);
        assert valid;
      }

      return valid;
    }
		
		@Override
		public Params clone() {
			Params p = new Params(stateCount, emissionCount);
			p.pi = pi.clone();
			p.T = T.clone();
			p.O = O.clone();
			
			return p;
		}

    public void updateFromVector(double[] weights) {
      assert( weights.length == numFeatures() );
      int idx = 0;
      for( int i = 0; i < stateCount; i++ ) {
        assert( idx == index_pi(i) );
        pi[i] = weights[idx++];
      }
      for( int i = 0; i < stateCount; i++ ) {
        for( int j = 0; j < stateCount; j++ ) {
          assert( idx == index_T(i,j) );
          T[i][j] = weights[idx++];
        }
      }
      for( int i = 0; i < stateCount; i++ ) {
        for( int j = 0; j < emissionCount; j++ ) {
           assert( idx == index_O(i,j) );
           O[i][j] = weights[idx++];
        }
      }
      assert( idx == numFeatures() );
      assert( isValid() );
    }
    public static Params fromVector(int stateCount, int emissionCount, double[] weights) {
      Params params = new Params( stateCount, emissionCount );
      int numFeatures = params.numFeatures();
      params.updateFromVector( weights );

      return params;
    }
		public double[] toVector() {
      double[] weights = new double[ numFeatures() ];
      int idx = 0;
      for( int i = 0; i < stateCount; i++ ) {
        //assert( idx == index_pi(i) );
        weights[idx++] = pi[i];
      }
      for( int i = 0; i < stateCount; i++ ) {
        for( int j = 0; j < stateCount; j++ ) {
         // assert( idx == index_T(i,j) );
          weights[idx++] = T[i][j];
        }
      }
      for( int i = 0; i < stateCount; i++ ) {
        for( int j = 0; j < emissionCount; j++ ) {
          //assert( idx == index_O(i,j) );
          weights[idx++] = O[i][j];
        }
      }
      assert( idx == numFeatures() );
      return weights;
		}
    public int numFeatures() {
      return stateCount +
             stateCount * stateCount +
             stateCount * emissionCount;
    }
    public int index_pi(int h){ 
      return h; 
    }
    public int index_T(int h, int h_){ 
      return stateCount + 
        stateCount * h + h_; 
    }
    public int index_O(int h, int d){ 
      return stateCount + 
        stateCount * stateCount + 
        emissionCount * h + d; 
    }
		
		public static Params uniformWithNoise(Random rand, int stateCount, int emissionCount, double noise) {
			Params p = new Params(stateCount, emissionCount);
			
			// Initialize each as uniform plus noise 
			// Pi
			for(int i = 0; i < stateCount; i++) {
				p.pi[i] = 1.0/stateCount;
				// Dividing the noise by the sqrt(size) so that the total noise is as given
				p.pi[i] += rand.nextGaussian() * Math.sqrt(noise/stateCount);
				// Ensure positive
				p.pi[i] = Math.abs( p.pi[i] );
			}
			MatrixOps.normalize( p.pi );
			
			// T
			for(int i = 0; i < stateCount; i++) {
				for(int j = 0; j < stateCount; j++) {
					p.T[i][j] = 1.0/stateCount;
					// Dividing the noise by the sqrt(size) so that the total noise is as given
					p.T[i][j] += rand.nextGaussian() * Math.sqrt(noise/stateCount);
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
					p.O[i][j] += rand.nextGaussian() * Math.sqrt(noise/emissionCount);
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
    @Option( gloss = "Generator for parameters" )
    Random paramRandom = new Random(1);
    @Option( gloss = "Generator for data" )
    Random genRandom = new Random(1);

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

  public final Params params;

  public HiddenMarkovModel( Params p ) {
    this.params = p;
  }
  public HiddenMarkovModel( int stateCount, int emissionCount ) {
		params = new Params( stateCount, emissionCount );
  }

  public static HiddenMarkovModel generate( GenerationOptions options ) {
		Params params = Params.uniformWithNoise( options.paramRandom, options.stateCount, options.emissionCount, options.noise );
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

		return Pair.with( observed, hidden );
	}

  /**
   * Generate from the model
   */
  @Deprecated
  public Pair<int[],int[]> generate(Random rnd, int length) {
    int[] observed = new int[length];
    int[] hidden = new int[length];

    int state = Multinomial.sample(rnd, params.pi);
    for(int i = 0; i < length; i++) {
      observed[i] = Multinomial.sample(rnd, params.O[state]);
      hidden[i] = state;
      state = Multinomial.sample(rnd, params.T[state]);
    }

    return Pair.with(observed, hidden);
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
		for( int s = 0; s < params.stateCount; s++ ) {
			// P( o_0 | s_k ) \pi(s_k)
			V[0][s] = params.O[s][o[0]] * params.pi[s];
			Ptr[0][s] = -1; // Doesn't need to be defined.
		}
    MatrixOps.normalize( V[0] );

		// The dynamic program to find the optimal path
		for( int i = 1; i < o.length; i++ ) {
			for( int s = 0; s < params.stateCount; s++ ) {
				// Find the max of T(s | s') V_(i-1)(s')
				double T_max = 0.0;
				int S_max = -1;
				for( int s_ = 0; s_ < params.stateCount; s_++ ) {
          double t = params.T[s_][s] * V[i-1][s_];
					if( t > T_max ) {
						T_max = t;
						S_max = s_;
					}
				}

				// P( o_i | s_k ) = P(o_i | s) *  max_j T(s | s') V_(i-1)(s')
				V[i][s] = params.O[s][o[i]] * T_max;
				Ptr[i][s] = S_max;
			}
      MatrixOps.normalize( V[i] );
		}

		int[] z = new int[o.length];
		// Choose the best last state and back track from there
		z[o.length-1] = MatrixOps.argmax(V[o.length-1]);
    if( z[o.length-1] == -1 ) {
      LogInfo.logs( Fmt.D( V ) );
      LogInfo.logs( Fmt.D( params.T ) );
    }
    assert( z[o.length-1] != -1 );
		for(int i = o.length-1; i >= 1; i-- )  {
      assert( z[i] != -1 );
			z[i-1] = Ptr[i][z[i]];
    }

		return z;
	}

	/**
	 * Use the forward-backward algorithm to find the posterior probability over states
	 * @param o
	 * @return
	 */
	public Pair<double[][],Double> forward( final int[] o ) {
		// Store the forward probabilities
		double [][] f = new double[o.length][params.stateCount];

    double c = 0;
    double c_ = 0;
		// Initialise with the initial probability
		for( int s = 0; s < params.stateCount; s++ ) {
			f[0][s] = params.pi[s] * params.O[s][o[0]];
		}
    c += Math.log( MatrixOps.sum( f[0] ) );
    MatrixOps.scale( f[0], 1/MatrixOps.sum(f[0]) );

		// Compute the forward values as f_t(s) = sum_{s_} f_{t-1}(s_) * T( s_ | s ) * O( y | s )
		for( int i = 1; i < o.length; i++ ) {
			for( int s = 0; s < params.stateCount; s++ ) {
				f[i][s] = 0.0;
				for( int s_ = 0; s_ < params.stateCount; s_++ ) {
					f[i][s] += params.T[s_][s] * f[i-1][s_];
				}
				f[i][s] *= params.O[s][o[i]];
			}
      double sum = MatrixOps.sum( f[i] );
      assert( sum < 1 );
      c_ = Math.log( sum );
      c += Math.log( sum );
      MatrixOps.scale( f[i], 1.0/sum );
    }
    assert( c < 0 );

    return new Pair<>(f,c);
	}
	public double[][] backward( final int[] o ) {
    // Backward probabilities
		double [][] b = new double[o.length][params.stateCount];
		for( int s = 0; s < params.stateCount; s++ ) {
			b[o.length-1][s] = 1.0;
		}
    MatrixOps.scale( b[o.length-1], 1/MatrixOps.sum(b[o.length-1]) );
		for( int i = o.length-2; i >= 0; i-- ) {
			for( int s = 0; s < params.stateCount; s++ ) {
				for( int s_ = 0; s_ < params.stateCount; s_++ ) {
					b[i][s] += b[i+1][s_] * params.T[s][s_] * params.O[s_][o[i+1]];
				}
			}
      MatrixOps.scale( b[i], 1/MatrixOps.sum(b[i]) );
		}

    return b;
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

  public void setParams(double[] params) {
    this.params.updateFromVector(params);
  }

  /**
   * Compute the expected log-likelihood.
   *
   */
  public double compute(int[][] X, double[] gradient) {
    double value = 0.0;
    int N = X.length;

    /** For each example, compute expected counts and update weights **/
    for( int n = 0; n < X.length; n++ ) {
      int[] o = X[n];
      assert(o.length > 0);

      Pair<double [][],Double> fc = forward( o );
      double f[][] = fc.getValue0();
      double c = fc.getValue1(); // the excess normalization constant.
      double b[][] = backward( o );

      double lhood_incr = Math.log(MatrixOps.sum(f[o.length-1])) + c;
      assert( lhood_incr < 0 );
      value += (lhood_incr - value)/(n+1);

      if( gradient != null ) {
        // Construct the transition probability
        double xi[][][] = new double[o.length][params.stateCount][params.stateCount];

        for( int t = 0; t < o.length-1; t++ ) {
          // Compute xi = P(h_t = i, h_{t+1} = j | O, \theta)
          //            = P(h_t = i | O, \theta) * P( h_{t+1} = j | h_t = i, O, \theta )
          //            = P(h_t = i | O, \theta) * P( h_{t+1} = j | h_t = i, O_{t+1}) * P( O | h_{t+1} ) / P(O)
          for( int i = 0; i < params.stateCount; i++ ) {
            for( int j = 0; j < params.stateCount; j++ ) {
              xi[t][i][j] = f[t][i] * params.T[i][j] * params.O[j][o[t+1]] * b[t+1][j]; // / norm;
            }
          }
          double norm_ = MatrixOps.sum(xi[t]);
          assert norm_ > 0;
          // Normalize
          MatrixOps.scale(xi[t], 1.0/norm_); //MatrixOps.sum(xi[t]));
        } {
          int t = o.length-1;
          for( int i = 0; i < params.stateCount; i++ ) {
            for( int j = 0; j < params.stateCount; j++ ) {
              xi[t][i][j] = f[t][i]; // / norm;
            }
          }
          double norm_ = MatrixOps.sum(xi[t]);
          assert norm_ > 0;
          // Normalize
          MatrixOps.scale(xi[t], 1.0/norm_);
        }

        {
          double norm = MatrixOps.sum(xi[0]);
          assert norm > 0;
          for( int h = 0; h < params.stateCount; h++ ) {
            gradient[params.index_pi(h)] += MatrixOps.sum(xi[0][h])/norm;
          }
//          double sum;
//          for( int h = 0; h < params.stateCount; h++ ) sum +=  gradient[params.index_pi(h)];
//          assert( MatrixOps.equal( sum, 1 ) );
        }

        //LogInfo.logs( "pi " + Fmt.D(gradient) );
        // update counts for T
        if( o.length > 1 ) {
          for( int h = 0; h < params.stateCount; h++ ) {
            double countH = 0.0;
            for( int t = 0; t < o.length-1; t++ )  countH += MatrixOps.sum( xi[t][h] ); // Number of times we're in state h.
            assert countH > 0;

            for( int h_ = 0; h_ < params.stateCount; h_++ ) {
              double countHH_ = 0; // Number of times we are going from h to h_
              for( int t = 0; t < o.length-1; t++ )  countHH_ += xi[t][h][h_];
              gradient[params.index_T(h,h_)] +=  countHH_/countH; // - gradient[params.index_T(h,h_)])/(n+1);
            }
//            double sum;
//            for( int h_ = 0; h_ < params.stateCount; h_++ ) sum +=  gradient[params.index_T(h,h_)];
//            assert( MatrixOps.equal( sum, 1 ) );
          }
        }
        //LogInfo.logs( "T " + Fmt.D(gradient) );
        // update counts for O
        for( int t = 0; t < o.length-1; t++ ) {
          double norm = MatrixOps.sum(xi[t]);
          assert norm > 0;
          for( int h = 0; h < params.stateCount; h++ ) {
            // Adding expected count that state $h$ produced observation $o$
            gradient[params.index_O(h,o[t])] += MatrixOps.sum(xi[t][h])/norm; // - gradient[params.index_O(h,o[t])]) / (n+1);
          }
        }
      }
    }
    // Normalize the gradient
    if( gradient != null ) {
      // Normalize pi
      double smoothing = 1e-6;
      {
        double norm = 0.0;
        for( int h = 0; h < params.stateCount; h++) {
          gradient[params.index_pi(h)] += smoothing;
          norm += gradient[params.index_pi(h)];
        }
        for( int h = 0; h < params.stateCount; h++) gradient[params.index_pi(h)] /= norm;
      }
      // Normalize T
      {
        for( int h = 0; h < params.stateCount; h++) {
          double norm = 0.0;
          for( int h_ = 0; h_ < params.stateCount; h_++) {
            gradient[params.index_T(h,h_)] += smoothing;
            norm += gradient[params.index_T(h,h_)];
          }
          for( int h_ = 0; h_ < params.stateCount; h_++) gradient[params.index_T(h,h_)] /= norm;
        }
      }
      // Normalize O
      {
        for( int h = 0; h < params.stateCount; h++) {
          double norm = 0.0;
          for( int o_ = 0; o_ < params.emissionCount; o_++) {
            gradient[params.index_O(h, o_)] += smoothing;
            norm += gradient[params.index_O(h, o_)];
          }
          for( int o_ = 0; o_ < params.emissionCount; o_++) gradient[params.index_O(h, o_)] /= norm;
        }
      }
    }
    //MatrixOps.scale( gradient, 1.0/N );
    //LogInfo.logs( "grad " + Fmt.D(gradient) );

    return value;
  }
  public double baumWelchStep(int[][] X) {
    double[] gradient = new double[ params.numFeatures() ];
    double llhood = compute(X, gradient);
    setParams(gradient);
    LogInfo.logs( Fmt.D( gradient ) );
    return llhood;
  }

  public int numFeatures() {
    return params.numFeatures();
  }

}

