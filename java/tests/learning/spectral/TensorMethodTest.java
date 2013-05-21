/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral;

import learning.linalg.*;

import learning.exceptions.NumericalException;
import learning.exceptions.RecoveryFailure;

import learning.models.MixtureOfGaussians;
import learning.models.MixtureOfGaussians.*;

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
public class TensorMethodTest {

  @Before 
  public void setUp() {
    //LogInfo.writeToStdout = false;
    LogInfo.init();
  }

  // Generation routines
  public MixtureOfGaussians generateMultiView(int K, int D, int V, MeanDistribution means) {
    return MixtureOfGaussians.generate(K, D, V, 
        WeightDistribution.Uniform, 
        means, 
        CovarianceDistribution.Spherical, 1.0);
  }
  public MixtureOfGaussians generateSmallEye() {
		int K = 2;
		int D = 3;
		int V = 3;
    return generateMultiView( K, D, V, MeanDistribution.Hypercube );
  }
  public MixtureOfGaussians generateSmallRandom() {
		int K = 2;
		int D = 3;
		int V = 3;
    return generateMultiView( K, D, V, MeanDistribution.Random );
  }
  public MixtureOfGaussians generateMediumEye() {
		int K = 4;
		int D = 6;
		int V = 3;
    return generateMultiView( K, D, V, MeanDistribution.Hypercube );
  }
  public MixtureOfGaussians generateMediumRandom() {
		int K = 4;
		int D = 6;
		int V = 3;
    return generateMultiView( K, D, V, MeanDistribution.Random );
  }

  // Actual tests

  public void testSymmetrization( MixtureOfGaussians model ) {
    int K = model.getK();
    int D = model.getD();
    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments = 
        model.computeExactMoments();

    TensorMethod algo = new TensorMethod();
    Pair<SimpleMatrix, FullTensor> symmetrizedMoments = TensorMethod.symmetrizeViews( 
        K,
        moments.getValue0(), // M12
        moments.getValue1(), // M13
        moments.getValue2(), // M23
        moments.getValue3() ); // M123
    // Test that these are symmetric indeed.
    SimpleMatrix Pairs = symmetrizedMoments.getValue0();
    FullTensor Triples = symmetrizedMoments.getValue1();

    Assert.assertTrue( Pairs.numRows() == D ) ; 
    Assert.assertTrue( Pairs.numCols() == D ) ; 
    Assert.assertTrue( Triples.D1 == D ) ; 
    Assert.assertTrue( Triples.D2 == D ) ; 
    Assert.assertTrue( Triples.D3 == D ) ; 

    Assert.assertTrue( MatrixOps.isSymmetric( Pairs ) );
    Assert.assertTrue( MatrixOps.isSymmetric( Triples ) );
  }

  @Test
  public void testSymmetrizationSmallEye() { testSymmetrization( generateSmallEye() ); }
  @Test
  public void testSymmetrizationSmallRandom() { testSymmetrization( generateSmallRandom() ); }
  @Test
  public void testSymmetrizationMediumEye() {  testSymmetrization( generateMediumEye() ); }
  @Test
  public void testSymmetrizationMediumRandom() {  testSymmetrization( generateMediumRandom() ); }

  public void testExactRunner( int K, int D, int V, MixtureOfGaussians model ) {
    // SimpleMatrix M3 = model.getMeans()[V-1];
    // SimpleMatrix[] M = model.getMeans();
    // Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments = 
    //     model.computeExactMoments();
    // try {
    //   TensorMethod algo = new TensorMethod();
    //   SimpleMatrix M3_ = algo.algorithmB( K, moments.getValue0(), moments.getValue1(), moments.getValue2() );
    //   M3_ = MatrixOps.alignMatrix( M3_, M3, true );

    //   Assert.assertTrue( MatrixOps.allclose( M3, M3_) );
    // } catch( RecoveryFailure e) {
    //   System.out.println( e.getMessage() );
    // }
  }
  // @Test
  public void testExact() {};

  public void testSampleRunner( int n, int K, int D, int V, MixtureOfGaussians model ) {
  }
  // @Test
  public void testSamples() {};

  @Option( gloss = "Number of points" )
  public double N = 1e4;
  
  public static void main( String[] args ) {
    TensorMethodTest test = new TensorMethodTest();
    MixtureOfGaussians.GenerationOptions genOptions = new MixtureOfGaussians.GenerationOptions();
    OptionsParser parser = new OptionsParser( test, genOptions );
    if( parser.parse( args ) ) {
    }
  }

}

