/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import learning.models.MixtureOfGaussians;
import static learning.models.MixtureOfGaussians.*;

import learning.linalg.MatrixOps;

import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

/**
 * Test code for various dimensions.
 */
public class MixtureOfGaussiansTest {

  // Generation routines
  public static MixtureOfGaussians generateMultiView(int K, int D, int V, MeanDistribution means) {
    return MixtureOfGaussians.generate(K, D, V, 
        WeightDistribution.Uniform, 
        means, 
        CovarianceDistribution.Spherical, 1.0);
  }
  public static MixtureOfGaussians generateSmallSymmetric() {
		int K = 2;
		int D = 3;
		int V = 3;
    return generateMultiView(K, D, V, MeanDistribution.Identical);
  }
  public static MixtureOfGaussians generateSmallEye() {
		int K = 2;
		int D = 3;
		int V = 3;
    return generateMultiView( K, D, V, MeanDistribution.Hypercube );
  }
  public static MixtureOfGaussians generateSmallRandom() {
		int K = 2;
		int D = 3;
		int V = 3;
    return generateMultiView( K, D, V, MeanDistribution.Random );
  }
  public static MixtureOfGaussians generateMediumSymmetric() {
		int K = 4;
		int D = 6;
		int V = 3;
    return generateMultiView( K, D, V, MeanDistribution.Identical );
  }
  public static MixtureOfGaussians generateMediumEye() {
		int K = 4;
		int D = 6;
		int V = 3;
    return generateMultiView( K, D, V, MeanDistribution.Hypercube );
  }
  public static MixtureOfGaussians generateMediumRandom() {
		int K = 4;
		int D = 6;
		int V = 3;
    return generateMultiView( K, D, V, MeanDistribution.Random );
  }
  public static MixtureOfGaussians generateSymmetricSparseEye() {
    int K = 2;
    int D = 8;
    int V = 3;
    return generateMultiView( K, D, V, MeanDistribution.Identical );
  }
  public static MixtureOfGaussians generateUnSymmetricSparseEye() {
    int K = 2;
    int D = 8;
    int V = 3;
    return generateMultiView( K, D, V, MeanDistribution.Hypercube );
  }

  public void testModel( int N, GenerationOptions options ) {
    int D = (int) options.D;
    int K = (int) options.K;
    int V = (int) options.V;

    MixtureOfGaussians model = MixtureOfGaussians.generate( options );
    SimpleMatrix[] X = model.sample( N );

    SimpleMatrix[] M = model.getMeans();

    for( int v = 0; v < V; v++ ) {
      int rnk = MatrixOps.rank( M[v] );
      Assert.assertTrue( rnk == K );
      Assert.assertTrue( X[v].numRows() == N );
      Assert.assertTrue( X[v].numCols() == D );
    }
  }
  public void testModel( GenerationOptions options ) {
    testModel( 1000, options );
  }

  @Test
  public void testDefault() {
    GenerationOptions options = new GenerationOptions();
    testModel( options );
  }

  @Test
  public void testRandomWeights() {
    GenerationOptions options = new GenerationOptions();
    options.weights = "random";
    testModel( options );
  }

  @Test
  public void testRandomMean() {
    GenerationOptions options = new GenerationOptions();
    options.means = "random";
    testModel( options );
  }

  @Test
  public void testSphericalCov() {
    GenerationOptions options = new GenerationOptions();
    options.covs = "spherical";
    testModel( options );
  }

  @Test
  public void testLarge() {
    int K = 6; int D = 20;

    GenerationOptions options = new GenerationOptions();
    options.K = K; options.D = D;
    testModel( options );
  }

}

