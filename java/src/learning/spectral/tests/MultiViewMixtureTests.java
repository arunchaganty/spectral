/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.tests;

import learning.models.MultiViewGaussianModel;
import learning.models.MultiViewGaussianModel.CovarianceDistribution;
import learning.models.MultiViewGaussianModel.MeanDistribution;
import learning.models.MultiViewGaussianModel.WeightDistribution;
import learning.spectral.MultiViewMixture;
import learning.spectral.MultiViewMixture.RecoveryFailure;
import learning.utils.MatrixFactory;
import learning.utils.MatrixOps;

import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
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
		
		try {
			SimpleMatrix M3_ = algo.exactRecovery(2, w, M1, M2, M3);
			System.out.println( M3 );
			System.out.println( M3_ );
		}
		catch (RecoveryFailure e) {
			Assert.fail();
		}
	}

	/**
	 * Test method for {@link learning.spectral.MultiViewMixture#sampleRecovery(int, org.ejml.simple.SimpleMatrix, org.ejml.simple.SimpleMatrix, org.ejml.simple.SimpleMatrix)}.
	 */
	@Test
	public void testSampleRecoverySimple() {
		MultiViewMixture algo = new MultiViewMixture();
		
		// Generate some random data 
		int k = 2;
		int d = 3;
		int V = 3;
		double[] wD = {0.4, 0.6};
		double[][] M1D = { {-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0} };
		double[][] M2D = { {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0} };
		double[][] M3D = { {1.0, 0.0, 1.0}, {0.0, -1.0, 1.0} };
		double[][][] M = {M1D,M2D,M3D};
		double[][][][] S = new double[V][k][][];
		for(int v = 0; v < V; v++ ) for(int i = 0; i<k; i++) S[v][i] = MatrixFactory.eye(d);
		
		MultiViewGaussianModel model = new MultiViewGaussianModel(k, d, V, wD, M, S);
		
		double[][][] X = model.sample( 100000 );
		
		try {
			SimpleMatrix M3_ = algo.sampleRecovery( k, X[0], X[1], X[2]);
			MatrixOps.printMatrix(M3D);
			System.out.println( M3_.transpose() );
		}
		catch (RecoveryFailure e) {
			Assert.fail();
		}
	}

	/**
	 * Test method for {@link learning.spectral.MultiViewMixture#sampleRecovery(int, org.ejml.simple.SimpleMatrix, org.ejml.simple.SimpleMatrix, org.ejml.simple.SimpleMatrix)}.
	 */
	@Test
	public void testSampleRecovery() {
		MultiViewMixture algo = new MultiViewMixture();
		
		// Generate some random data 
		int k = 4;
		int d = 6;
		int V = 3;
		
		MultiViewGaussianModel model = MultiViewGaussianModel.generate(k, d, V, 1.0, WeightDistribution.Uniform, MeanDistribution.Hypercube, CovarianceDistribution.Spherical);
		
		double[][][] X = model.sample( 100000 );
		
		try {
			SimpleMatrix M3_ = algo.sampleRecovery( k, X[0], X[1], X[2]);
			MatrixOps.printMatrix( model.getM()[2] );
			System.out.println( M3_.transpose() );
		}
		catch (RecoveryFailure e) {
			Assert.fail();
		}
	}

}
