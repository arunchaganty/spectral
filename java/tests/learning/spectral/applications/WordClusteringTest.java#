package learning.spectral.tests;

import learning.models.HMM;
import learning.spectral.MultiViewMixture;
import learning.spectral.MultiViewMixture.RecoveryFailure;

import org.ejml.simple.SimpleMatrix;
import org.junit.Test;

public class WordClusteringTest {

	@Test
	public void testHMMParameterRecovery() throws RecoveryFailure {
		int k = 2;
		int d = 3;
		int N = (int) 1e6;
		double[][] features = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
		
		// Create a simple low-dimensional HMM
		HMM.Params params = HMM.Params.uniformWithNoise(k, d, 0.1);
		HMM hmm = new HMM(params);
		
		
		// Generate some data
		double[][] X1 = new double[N][2];
		double[][] X2 = new double[N][2];
		double[][] X3 = new double[N][2];
		for( int i = 0; i < N; i++ ) 
		{
			int[] x = hmm.generateObservationSequence(3);
			X1[i] = features[x[0]];
			X2[i] = features[x[1]];
			X3[i] = features[x[2]];
		}
		
		// Test whether the multi-view method can recover the parameters
		MultiViewMixture algo = new MultiViewMixture();
		
		SimpleMatrix O = (new SimpleMatrix(params.O)).transpose();
		SimpleMatrix T = (new SimpleMatrix(params.T));
		SimpleMatrix O_ = algo.sampleRecovery(k, X3, X1, X2 );
		SimpleMatrix OT_ = algo.sampleRecovery(k, X1, X2, X3 );
		
		System.out.println( O );
		System.out.println( O_ );
		System.out.println( O.mult(T) );
		System.out.println( OT_ );
		
		
	}

}
