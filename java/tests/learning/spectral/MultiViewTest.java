/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral;

import learning.linalg.MatrixOps;
import learning.linalg.Tensor;

//import learning.spectral.MultiViewMixtures;
import learning.exceptions.NumericalException;
import learning.exceptions.RecoveryFailure;

import learning.models.MixtureOfGaussians;
import learning.models.MixtureOfGaussians.*;

import org.javatuples.*;

import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

/**
 * 
 */
public class MultiViewTest {

  public void testAlgorithmBExact( int K, int D, int V, MixtureOfGaussians model ) {
    SimpleMatrix M3 = model.getMeans()[V-1];

		MultiViewMixture algo = new MultiViewMixture();
    Triplet<SimpleMatrix, SimpleMatrix, Tensor> moments = algo.computeExactMoments( D, K, V, model.getWeights(), model.getMeans() );
    try {
      SimpleMatrix M3_ = algo.algorithmB( K, moments.getValue0(), moments.getValue1(), moments.getValue2() );
      M3_ = MatrixOps.alignMatrix( M3_, M3 );

      Assert.assertTrue( MatrixOps.allclose( M3, M3_) );
    } catch( RecoveryFailure e) {
      System.out.println( e.getMessage() );
    }
  }

  @Test
  /**
   * Test algorithmB with exact moments
   */
  public void algorithmBExactSmall() {
		int K = 2;
		int D = 3;
		int V = 3;
		
		MixtureOfGaussians model = MixtureOfGaussians.generate(K, D, V, WeightDistribution.Uniform, MeanDistribution.Hypercube, CovarianceDistribution.Spherical, 1.0);

    testAlgorithmBExact( K, D, V, model );
  }

  @Test
  /**
   * Test algorithmB with exact moments
   */
  public void algorithmBExactMedium() {
		int K = 4;
		int D = 6;
		int V = 3;
		
		MixtureOfGaussians model = MixtureOfGaussians.generate(K, D, V, WeightDistribution.Uniform, MeanDistribution.Hypercube, CovarianceDistribution.Spherical, 1.0);

    testAlgorithmBExact( K, D, V, model );
  }

  @Test
  /**
   * Test algorithmB with exact moments
   */
  public void algorithmBExactLarge() {
		int K = 30;
		int D = 100;
		int V = 3;
		
		MixtureOfGaussians model = MixtureOfGaussians.generate(K, D, V, WeightDistribution.Uniform, MeanDistribution.Hypercube, CovarianceDistribution.Spherical, 1.0);

    testAlgorithmBExact( K, D, V, model );
  }

  public void testAlgorithmB( int N, int K, int D, int V, MixtureOfGaussians model ) {
    SimpleMatrix M3 = model.getMeans()[V-1];

    SimpleMatrix[] X = model.sample( N );

		MultiViewMixture algo = new MultiViewMixture();
    try {
      SimpleMatrix M3_ = algo.recoverM3( K, X[0], X[1], X[2] );
      M3_ = MatrixOps.alignMatrix( M3_, M3 );

      System.out.println( M3_ );
      System.out.println( M3 );

      double err = MatrixOps.norm( M3.minus( M3_ ) );
      System.out.println( err );

      Assert.assertTrue( MatrixOps.allclose( M3, M3_, 1e-1 ) );
    } catch( NumericalException | RecoveryFailure e) {
      System.out.println( e.getMessage() );
    }
  }


  @Test
  /**
   * Test algorithmB with exact moments
   */
  public void algorithmBSmall() {
		int N = (int) 1e4;
		int K = 2;
		int D = 3;
		int V = 3;
		
		MixtureOfGaussians model = MixtureOfGaussians.generate(K, D, V, WeightDistribution.Uniform, MeanDistribution.Hypercube, CovarianceDistribution.Spherical, 1.0);

    testAlgorithmB( N, K, D, V, model );
  }

  @Test
  /**
   * Test algorithmB with exact moments
   */
  public void algorithmBMedium() {
		int N = (int) 1e5;
		int K = 4;
		int D = 6;
		int V = 3;
		
		MixtureOfGaussians model = MixtureOfGaussians.generate(K, D, V, WeightDistribution.Uniform, MeanDistribution.Hypercube, CovarianceDistribution.Spherical, 1.0);

    testAlgorithmB( N, K, D, V, model );
  }

  @Test
  /**
   * Test algorithmB with exact moments
   */
  public void algorithmBLarge() {
		int N = (int) 1e6;
		int K = 30;
		int D = 50;
		int V = 3;
		
		MixtureOfGaussians model = MixtureOfGaussians.generate(K, D, V, WeightDistribution.Uniform, MeanDistribution.Hypercube, CovarianceDistribution.Spherical, 1.0);

    testAlgorithmB( N, K, D, V, model );
  }

  

  public static void main( String[] args ) {
    MultiViewTest test = new MultiViewTest();
    test.algorithmBSmall();
  }

}
