/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Vector;

import learning.utils.RandomFactory;

/**
 * A HMM model for sentences
 */
public class SentenceHMM implements Serializable {
	private static final long serialVersionUID = -7983543656788035850L;
	
	/// HMM parameters
	public int N; // Number of states
	public double[] pi; // Initial distribution
	public double[][] T; // Transition matrix
	// Storing the observation matrix as a sparse map (only choose the top 1000 words for each group)
	public double[][] O; // Observation matrix
	public int[][] words; // Word mapping for the observation matrix
	public String[] dict; // State -> String mapping
	
	public SentenceHMM( double[] pi, double[][] O, int[][] words, double[][] T, String[] dict )
	{
		this.N = pi.length;
		this.pi = pi;
		this.O = O;
		this.words = words;
		this.T = T;
		this.dict = dict;
	}
	
	/**
	 * Generate a single string of length 'n' from the HMM model
	 * @param n
	 * @return
	 */
	public String generateString(int n)
	{
		String output = "";
		// Pick a start state
		int state = RandomFactory.multinomial(pi);
		
		for( int i = 0; i < n; i++)
		{
			// Generate a word
			int o = RandomFactory.multinomial( O[state] );
			output += dict[words[state][o]] + " ";
			// Transit to a new state
			state = RandomFactory.multinomial( T[state] );
		}
		
		output.trim();
		
		return output;
	}
	
	protected static class IndexedComparator implements Comparator<Integer> {
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
	 * Learn the parameters for a sentence HMM given the sequence of X's (observed variables) and Z's (latent variables)
	 * @param emissionDim
	 * @param latentDim
	 * @param X
	 * @param O
	 * @param dict
	 * @return
	 */
	public static SentenceHMM learnFullyObserved( int emissionDim, int latentDim, int wordsPerState, int[][] X, int[][] Z, String[] dict )
	{
		assert( X.length == Z.length );
		int N = X.length;
		
		double[] pi = new double[ latentDim ];
		int piCount = 0;
		double[][] T = new double[ latentDim ][latentDim];
		int[] TCount = new int[ latentDim ];
		final double[][] O = new double[ latentDim ][emissionDim];
		int[] OCount = new int[ latentDim ];
		
		// Initialise with some simple smoothing
		for( int i = 0; i < latentDim; i++ )
			pi[i] = 1.0/latentDim;
		piCount = latentDim;
		
		for( int i = 0; i < latentDim; i++ )
		{
			for( int j = 0; j < latentDim; j++ )
				T[i][j] = 1.0/latentDim;
			TCount[i] = latentDim;
		}
		
		for( int i = 0; i < latentDim; i++ )
		{
			for( int j = 0; j < emissionDim; j++ )
				O[i][j] = 1.0/emissionDim;
			OCount[i] = emissionDim;
		}
		
		// For each sequence in X and Z, increment those counts (do simple smoothing)
		for( int i = 0; i < N; i++ ) {
			pi[Z[i][0]] += (1 - pi[Z[i][0]])/(++piCount);
			for( int j = 0; j < Z[i].length; j++ )
			{
				if( j < Z[i].length - 1)
					T[Z[i][j]][Z[i][j+1]] += (1 - T[Z[i][j]][Z[i][j+1]])/(++TCount[Z[i][j]]);
				O[Z[i][j]][X[i][j]] += (1 - O[Z[i][j]][X[i][j]])/(++OCount[Z[i][j]]);
			}
		}
		
		// Compress
		double[][] O_ = new double[latentDim][wordsPerState];
		int[][] words = new int[latentDim][wordsPerState];
		for( int i = 0; i < latentDim; i++ ) {
			Integer[] words_ = new Integer[emissionDim];
			for(int j = 0; j<emissionDim; j++) words_[j] = j;
			
			// Choose top k words
			Arrays.sort(words_, new SentenceHMM.IndexedComparator(O[i]) );
			for( int j = 0; j < wordsPerState; j++ ) {
				O_[i][j] = O[i][words_[j]];
				words[i][j] = words_[j];
			}
		}
			
		return new SentenceHMM(pi, O_, words, T, dict);
	}

}
