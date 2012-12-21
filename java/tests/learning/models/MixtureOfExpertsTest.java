/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import learning.models.MixtureOfExperts;
import learning.models.MixtureOfExperts.GenerationOptions;

import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

/**
 * Test code for various dimensions.
 */
public class MixtureOfExpertsTest {

  public void testModel( int N, GenerationOptions options ) {
    int D = (int) options.D;

    MixtureOfExperts model = MixtureOfExperts.generate( options );
    SimpleMatrix[] data = model.sample( N );

    SimpleMatrix y = data[0];
    SimpleMatrix X = data[1];
    Assert.assertTrue( y.numCols() == N );
    Assert.assertTrue( X.numRows() == N );
    Assert.assertTrue( X.numCols() == D );

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
  public void testDefaultWithNoise() {
    GenerationOptions options = new GenerationOptions();
    options.sigma2 = 1.0;

    testModel( options );
  }

  @Test
  public void testRandomWeights() {
    GenerationOptions options = new GenerationOptions();
    options.weights = "random";
    testModel( options );
  }

  @Test
  public void testRandomBeta() {
    GenerationOptions options = new GenerationOptions();
    options.betas = "random";
    testModel( options );
  }

  @Test
  public void testRandomMean() {
    GenerationOptions options = new GenerationOptions();
    options.mean = "random";
    testModel( options );
  }

  @Test
  public void testSphericalCov() {
    GenerationOptions options = new GenerationOptions();
    options.cov = "spherical";
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

