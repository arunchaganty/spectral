/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import learning.models.MixtureOfGaussians;
import learning.models.MixtureOfGaussians.GenerationOptions;

import learning.linalg.MatrixOps;

import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

/**
 * Test code for various dimensions.
 */
public class MixtureOfGaussiansTest {

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

