/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import learning.models.MixtureOfExperts;
import learning.models.MixtureOfExperts.GenerationOptions;
import learning.models.transforms.NonLinearity;

import learning.linalg.MatrixOps;

import learning.models.transforms.PolynomialNonLinearity;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.*;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

/**
 * Test code for various dimensions.
 */
public class MixtureOfExpertsTest {

  @Test
  public void testNonLinearUnitGeneration() {
    int D = 3;
    double x[] = {1.0, 2.0, 3.0};

    NonLinearity nl = new PolynomialNonLinearity();
    Assert.assertTrue( nl.getLinearDimension( D ) == D );
    double[] y = nl.getLinearEmbedding( x );
    for( int i = 0; i < D; i++ )
      Assert.assertTrue( x[i] == y[i] );

    nl = new PolynomialNonLinearity(2);
    double z[] = {1.0, 2.0, 3.0, 4.0, 6.0, 9.0 };
    Assert.assertTrue( nl.getLinearDimension( D ) == z.length );
    y = nl.getLinearEmbedding( x );
    for( int i = 0; i < D; i++ )
      Assert.assertTrue( y[i] == z[i] );
  }

  public void testModel( int N, GenerationOptions options ) {
    int D = (int) options.D;

    MixtureOfExperts model = MixtureOfExperts.generate( options );
    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample( N );


    int D_ = model.getNonLinearity().getLinearDimension( D +
        (options.bias ? 1 : 0) );

    SimpleMatrix y = yX.getValue0();
    SimpleMatrix X = yX.getValue1();
    Assert.assertTrue( y.numCols() == N );
    Assert.assertTrue( X.numRows() == N );
    Assert.assertTrue( X.numCols() == D_ );
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
  public void testRandomNoBias() {
    GenerationOptions options = new GenerationOptions();
    options.weights = "random";
    options.bias = false;
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
  public void testNonLinear() {
    GenerationOptions options = new GenerationOptions();
    options.nlDegree = 2;
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

