
package learning.models;

import java.io.Serializable;
import java.util.Arrays;

import learning.utils.Misc;
import learning.utils.RandomFactory;

/**
 * The Hidden Markov Model class
 */
public class HMM implements Serializable {
	private static final long serialVersionUID = -6258851600180414061L;

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
			Misc.renormalize( p.pi );
			
			// T
			for(int i = 0; i < stateCount; i++) {
				for(int j = 0; j < stateCount; j++) {
					p.T[i][j] = 1.0/stateCount;
					// Dividing the noise by the sqrt(size) so that the total noise is as given
					p.T[i][j] += RandomFactory.randn( noise / Math.sqrt(stateCount) );
					// Ensure positive
					p.T[i][j] = Math.abs( p.T[i][j] );
				}
				Misc.renormalize( p.T[i] );
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
				Misc.renormalize( p.O[i] );
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
			
			// Renormalize
			Misc.renormalize( p.pi );
			for( int i = 0; i < stateCount; i++ ) {
				Misc.renormalize( p.O[i] );
				Misc.renormalize( p.T[i] );
			}
			
			return p;
		}
	}

	protected Params params;
	
	public HMM(int stateCount, int emissionCount ) {
		params = new Params( stateCount, emissionCount );
	}
	public HMM(Params p) {
		params = p;
	}
	
	/**
	 * Generate a single observation sequence of length n
	 * @param n
	 * @return
	 */
	public int[] generateObservationSequence(int n)
	{
		int[] output = new int[n];
		
		// Pick a start state
		int state = RandomFactory.multinomial(params.pi);
		
		for( int i = 0; i < n; i++)
		{
			// Generate a word
			int o = RandomFactory.multinomial( params.O[state] );
			output[i] = params.map[state][o];
			// Transit to a new state
			state = RandomFactory.multinomial( params.T[state] );
		}
		
		return output;
	}
	
	/**
	
	/**
	 * Use the Viterbi dynamic programming algorithm to find the hidden states for o.
	 * @param o
	 * @return
	 */
	public int[] viterbi( int[] o ) {
		return null;
	}
	
	public static HMM learnFullyObserved( int stateCount, int emissionCount, int[][] X, int[][] Z, 
			boolean shouldSmooth) {
		Params p = Params.fromCounts( stateCount, emissionCount, X, Z, shouldSmooth);
		
		return new HMM(p);
	}	
		
	public static HMM learnFullyObserved( int stateCount, int emissionCount, int[][] X, int[][] Z, 
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
			Misc.renormalize( O_[i] );
		}
		
		Params p_ = new Params(stateCount, compressedEmissionCount);
		p_.pi = p.pi; p_.T = p.T; p_.O = O_; p_.map = map;
		
		return new HMM(p_);
	}

}
