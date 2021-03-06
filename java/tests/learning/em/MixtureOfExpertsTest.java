/**
 * learning.em
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.em;

import learning.exceptions.NumericalException;
import learning.exceptions.RecoveryFailure;

import learning.em.MixtureOfExperts;
import learning.em.MixtureOfExperts.*;
import learning.models.MixtureOfExperts.*;

import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;

import org.javatuples.*;

import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.basic.LogInfo;

/**
 * 
 */
public class MixtureOfExpertsTest {

  @Before 
  public void setUp() {
    LogInfo.writeToStdout = false;
    LogInfo.init();
  }

  public void testRecovery( int N, int K, int D, learning.models.MixtureOfExperts model ) {
    SimpleMatrix betas = model.getBetas().transpose();
    SimpleMatrix weights = model.getWeights();

    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample( N );
    SimpleMatrix y = yX.getValue0();
    SimpleMatrix X = yX.getValue1();

    MixtureOfExperts algo = new MixtureOfExperts( K );

    Parameters params = algo.run( y, X );

    SimpleMatrix weights_ = MatrixOps.alignMatrix( MatrixFactory.fromVector( params.weights ), weights );
    double weight_err = MatrixOps.norm( weights.minus( weights_ ) );

    SimpleMatrix betas_ = new SimpleMatrix( params.betas );
    betas_ = MatrixOps.alignMatrix( betas_, betas );
    double beta_err = MatrixOps.norm( betas.minus( betas_ ) );
    System.err.printf( "weights: %f, betas: %f\n", weight_err, beta_err );

    Assert.assertTrue( MatrixOps.allclose( betas, betas_, 1e-1 ) );
    Assert.assertTrue( MatrixOps.allclose( weights, weights_, 1e-1 ) );
  }


  @Test
  /**
   * Test algorithmB with exact moments
   */
  public void testSmall() {
		int N = (int) 1e4;
		int K = 2;
		int D = 3;
		double sigma2 = 0.1;
    learning.models.MixtureOfExperts model = learning.models.MixtureOfExperts.generate( K, D, sigma2, WeightDistribution.Uniform, BetaDistribution.Eye, MeanDistribution.Zero, CovarianceDistribution.Eye );
		
    testRecovery( N, K, D, model );
  }

  @Test
  /**
   * Test algorithmB with exact moments
   */
  public void testMedium() {
		int N = (int) 1e5;
		int K = 6;
		int D = 10;
		double sigma2 = 0.1;
    learning.models.MixtureOfExperts model = learning.models.MixtureOfExperts.generate( K, D, sigma2, WeightDistribution.Uniform, BetaDistribution.Eye, MeanDistribution.Zero, CovarianceDistribution.Eye );
		
    testRecovery( N, K, D, model );
  }

  @Test
  /**
   * Test algorithmB with exact moments
   */
  public void testNoisy() {
		int N = (int) 1e5;
		int K = 3;
		int D = 5;
		double sigma2 = 0.6;
    learning.models.MixtureOfExperts model = learning.models.MixtureOfExperts.generate( K, D, sigma2, WeightDistribution.Uniform, BetaDistribution.Eye, MeanDistribution.Zero, CovarianceDistribution.Eye );
		
    testRecovery( N, K, D, model );
  }

  @Test
  /**
   * Test algorithmB with exact moments
   */
  public void testRandomBeta() {
		int N = (int) 1e5;
		int K = 3;
		int D = 5;
		double sigma2 = 0.2;
    learning.models.MixtureOfExperts model = learning.models.MixtureOfExperts.generate( K, D, sigma2, WeightDistribution.Random, BetaDistribution.Random, MeanDistribution.Zero, CovarianceDistribution.Eye );
		
    testRecovery( N, K, D, model );
  }

  @Test
  /**
   * Test algorithmB with exact moments
   */
  public void testRandomData() {
		int N = (int) 1e5;
		int K = 3;
		int D = 5;
		double sigma2 = 0.2;
    learning.models.MixtureOfExperts model = learning.models.MixtureOfExperts.generate( K, D, sigma2, WeightDistribution.Uniform, BetaDistribution.Eye, MeanDistribution.Random, CovarianceDistribution.Spherical );
		
    testRecovery( N, K, D, model );
  }

  /**
   * Test algorithmB with exact moments
   */
  public void arbitraryTest() {
    learning.models.MixtureOfExperts model = learning.models.MixtureOfExperts.generate( K, D, sigma2, WeightDistribution.Uniform, BetaDistribution.Eye, MeanDistribution.Zero, CovarianceDistribution.Eye );
    testRecovery( (int) N, K, D, model );
  }

  @Option( gloss = "Number of points" )
  public double N = 1e4;
  @Option( gloss = "Number of clusters" )
  public int K = 2;
  @Option( gloss = "Number of dimensions" )
  public int D = 3;
  @Option( gloss = "Covariance parameter" )
  public double sigma2 = 0.1;
  
  public static void main( String[] args ) {
    MixtureOfExpertsTest test = new MixtureOfExpertsTest();
    OptionsParser parser = new OptionsParser( test );

    if( parser.parse( args ) ) {
      test.arbitraryTest();
    }
  }
}

