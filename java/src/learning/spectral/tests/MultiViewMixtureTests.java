/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.tests;

import static org.junit.Assert.*;
import learning.models.MultiViewGaussianModel;
import learning.spectral.MultiViewMixture;

import org.ejml.simple.SimpleMatrix;
import org.junit.Test;

/**
 * 
 */
public class MultiViewMixtureTests {

	/**
	 * Test method for {@link learning.spectral.MultiViewMixture#exactRecovery(int, org.ejml.simple.SimpleMatrix, org.ejml.simple.SimpleMatrix, org.ejml.simple.SimpleMatrix, org.ejml.simple.SimpleMatrix)}.
	 */
	@Test
	public void testExactRecovery() {
		MultiViewMixture algo = new MultiViewMixture();
		
		// Generate some random data 
		double[][] wD = { {0.4, 0.6} };
		double[][] M1D = { {-1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0} };
		double[][] M2D = { {-1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0} };
		double[][] M3D = { {-1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0} };
		
		SimpleMatrix w = new SimpleMatrix( wD );
		SimpleMatrix M1 = new SimpleMatrix( M1D );
		SimpleMatrix M2 = new SimpleMatrix( M2D );
		SimpleMatrix M3 = new SimpleMatrix( M3D );
		
		SimpleMatrix M3_ = algo.exactRecovery(2, w, M1, M2, M3);
		System.out.println( M3 );
		System.out.println( M3_ );
	}

	/**
	 * Test method for {@link learning.spectral.MultiViewMixture#sampleRecovery(int, org.ejml.simple.SimpleMatrix, org.ejml.simple.SimpleMatrix, org.ejml.simple.SimpleMatrix)}.
	 */
	@Test
	public void testSampleRecovery() {
		MultiViewMixture algo = new MultiViewMixture();
		
		// Generate some random data 
		int k = 2;
		int d = 3;
		int V = 3;
		double[][] wD = { {0.4, 0.6} };
		double[][] M1D = { {-1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0} };
		double[][] M2D = { {-1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0} };
		double[][] M3D = { {-1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0} };
		
		SimpleMatrix w = new SimpleMatrix( wD );
		SimpleMatrix M1 = new SimpleMatrix( M1D );
		SimpleMatrix M2 = new SimpleMatrix( M2D );
		SimpleMatrix M3 = new SimpleMatrix( M3D );
		SimpleMatrix[] M = {M1,M2,M3};
		
		MultiViewGaussianModel model = new MultiViewGaussianModel(k, d, V, w, M, null);
		
		SimpleMatrix[] X = model.sample( 100000 );
		
		SimpleMatrix M3_ = algo.sampleRecovery( k, X[0], X[1], X[2]);
		System.out.println( M3 );
		System.out.println( M3_ );
	}

}
