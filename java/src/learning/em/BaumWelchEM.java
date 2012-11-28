/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.em;

import learning.models.HMM;
import learning.utils.Corpus;

/**
 * Baum Welch EM
 */
public class BaumWelchEM {
	double noise = 0.1;
	double maxIters = 100;
	
	public HMM learn( int stateCount, int emissionCount, int[][] X, boolean shouldSmooth ) {
		HMM.Params params = HMM.Params.uniformWithNoise(stateCount, emissionCount, noise);
		return learn( params, X, shouldSmooth );
	}
	
	public HMM learn( HMM.Params initialParams, int[][] X, boolean shouldSmooth ) {
		// From the initial state, parse all the sentences X and then find the ML parameters, and repeat
		HMM hmm = new HMM(initialParams);
		HMM.Params p = initialParams.clone();
		int stateCount = initialParams.stateCount;
		int emissionCount = initialParams.emissionCount;
		
		for( int i = 0; i < maxIters; i++ ) {
			// Use Viterbi to populate all the observed states
			int[][] Z = new int[X.length][];
			for( int j = 0; j < X.length; j++ ) {
				Z[j] = hmm.viterbi(X[j]);
			}
			
			// Learn the parameters using ML
			HMM.Params p_ = HMM.Params.fromCounts(stateCount, emissionCount, X, Z, shouldSmooth);
			
			// Check for convergence in terms of KL divergence
			p = p_;
		}
		
		return new HMM(p);
	}
	
	/**
	 * Learn the parameters for a sentence HMM given the sequence of X's (observed variables).
	 * @param C
	 * @param wordsPerState
	 * @return
	 */
	public HMM learn( Corpus C, int wordsPerState ) {
		return null;
	}

}
