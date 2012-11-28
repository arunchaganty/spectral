/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import java.io.Serializable;
import learning.utils.ParsedCorpus;

/**
 * A HMM model for sentences
 */
public class SentenceHMM extends HMM implements Serializable {
	private static final long serialVersionUID = -7983543656788035850L;
	
	public String[] dict; // State -> String mapping
	
	public SentenceHMM( Params p, String[] dict ) {
		super( p );
		this.dict = dict;
	}
	
	/**
	 * Generate a single string of length 'n' from the HMM model
	 * @param n
	 * @return
	 */
	public String generateString(int n)
	{
		int[] output_ = generateObservationSequence(n);
		String output = "";
		
		for( int i = 0; i < n; i++)
			output += dict[output_[i]] + " ";
		output.trim();
		
		return output;
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
	public static SentenceHMM learnFullyObserved( ParsedCorpus C, int wordsPerState, boolean shouldSmooth ) {
		int[][] X = C.C;
		int[][] Z = C.Z;
		int latentDim = C.Zdict.length;
		int emissionDim = C.dict.length;
		
		HMM hmm = HMM.learnFullyObserved(latentDim, emissionDim, X, Z, shouldSmooth, wordsPerState);
			
		return new SentenceHMM(hmm.params, C.dict );
	}
}
