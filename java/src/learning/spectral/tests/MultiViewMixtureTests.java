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

import fig.basic.LogInfo;

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
		double[] w = {0.4, 0.6};
		double[][] M1 = { {-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0} };
		double[][] M2 = { {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0} };
		double[][] M3 = { {1.0, 0.0, 1.0}, {0.0, -1.0, 1.0} };
		double[][][] M = {M1,M2,M3};
		double[][][][] S = new double[V][k][][];
		for(int v = 0; v < V; v++ ) for(int i = 0; i<k; i++) S[v][i] = MatrixFactory.eye(d);
		
		MultiViewGaussianModel model = new MultiViewGaussianModel(k, d, V, w, M, S);
		
		double[][][] X = model.sample( 1000000 );
		
		try {
			double[][] M3_ = algo.sampleRecovery( k, X[0], X[1], X[2]);
      M3_ = MatrixOps.alignMatrix( M3_, M3, false ); 
      MatrixOps.printMatrix( M3 );
      MatrixOps.printMatrix( M3_ );
      double err = MatrixOps.norm( MatrixOps.matrixSub( M3, M3_ ) );
      double rerr = err/MatrixOps.norm( M3 );
		  LogInfo.logsForce( "abs-err: " + err );
		  LogInfo.logsForce( "rel-err: " + rerr );
      if( err > 1e-2 )
			  Assert.fail();
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
    double[][] M3 = model.getM()[2];
		
		double[][][] X = model.sample( 1000000 );
		
		try {
			double[][] M3_ = algo.sampleRecovery( k, X[0], X[1], X[2]);
      M3_ = MatrixOps.alignMatrix( M3_, M3, false ); 
      MatrixOps.printMatrix( M3 );
      MatrixOps.printMatrix( M3_ );
      double err = MatrixOps.norm( MatrixOps.matrixSub( M3, M3_ ) );
      double rerr = err/MatrixOps.norm( M3 );
		  LogInfo.logsForce( "abs-err: " + err );
		  LogInfo.logsForce( "rel-err: " + rerr );
      if( err > 1e-2 )
			  Assert.fail();
		}
		catch (RecoveryFailure e) {
			Assert.fail();
		}
	}

  public static void main(String[] args) {
	  //testExactRecovery();
	  //testSampleRecoverySimple();
	  //testSampleRecovery();
  }

}
