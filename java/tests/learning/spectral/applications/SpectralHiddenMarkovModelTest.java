/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.Tensor;
import learning.exceptions.NumericalException;

import learning.models.RealHiddenMarkovModel;
import learning.models.RealHiddenMarkovModel.*;
import learning.models.HiddenMarkovModel.*;

import org.javatuples.*;
import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

import fig.basic.LogInfo;

/**
 * 
 */
public class SpectralHiddenMarkovModelTest {

  @Before 
  public void setUp() {
    LogInfo.writeToStdout = false;
    LogInfo.init();
  }

  public void testMoments(int K, int D, int N, int M, RealHiddenMarkovModel model) {
    Triplet<SimpleMatrix, SimpleMatrix, Tensor> moments = SpectralHiddenMarkovModel.computeExactMoments( model );

    double[][][] data = model.sampleReal( N, M );

    SpectralHiddenMarkovModel algo = new SpectralHiddenMarkovModel(K, D);
    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix[], SimpleMatrix> moments_ = algo.partitionData( data );

    double err = 0.0;

    SimpleMatrix P12 = moments.getValue0();
    SimpleMatrix P12_ = moments_.getValue0();
    err = MatrixOps.norm( P12.minus( P12_ ) );
    System.err.println( "P12: " + err );
    Assert.assertTrue( MatrixOps.allclose( P12, P12_, 1e-2 ) );

    SimpleMatrix P13 = moments.getValue1();
    SimpleMatrix P13_ = moments_.getValue1();
    err = MatrixOps.norm( P13.minus( P13_ ) );
    System.err.println( "P13: " + err );
    Assert.assertTrue( MatrixOps.allclose( P13, P13_, 1e-2 ) );

    SimpleMatrix theta = moments_.getValue3();
    SimpleMatrix U2 = MatrixOps.svdk( P12_, K )[2];
    SimpleMatrix Theta = U2.mult( theta );

    for( int k = 0; k < K; k++ ) {
      SimpleMatrix P132T = moments.getValue2().project( 1, MatrixOps.col( Theta, k ) );
      SimpleMatrix P132T_ = moments_.getValue2()[k];
      err = MatrixOps.norm( P132T.minus( P132T_ ) );
      System.err.printf( "P132T[%d]: %f\n", k, err );
      Assert.assertTrue( MatrixOps.allclose( P132T, P132T_, 1e-2 ) );
    }
  }

  @Test
  public void testMoments() throws NumericalException {
    int N = (int) 1e6;
    int M = (int) 3;

    int K = 2;
    int E = 3;
    int D = 3;

    GenerationOptions genOptions = new GenerationOptions(K, E);
    FeatureOptions featureOptions = new FeatureOptions(D, "eye");

    RealHiddenMarkovModel model = RealHiddenMarkovModel.generate( genOptions, featureOptions );
    testMoments( K, D, N, M, model );
  }

  @Test
  public void testMomentsWithNoise() throws NumericalException {
    int N = (int) 1e6;
    int M = (int) 3;

    int K = 2;
    int E = 3;
    int D = 3;
    double noise = 1e-1;

    GenerationOptions genOptions = new GenerationOptions(K, E);
    FeatureOptions featureOptions = new FeatureOptions(D, "eye", noise);

    RealHiddenMarkovModel model = RealHiddenMarkovModel.generate( genOptions, featureOptions );
    testMoments( K, D, N, M, model );
  }

  public void testHiddenMarkovModel(int N, int M, RealHiddenMarkovModel model) 
    throws NumericalException {
    int K = model.getStateCount();
    int D = model.getDimension();

    SimpleMatrix O = model.getRealO();

    double[][][] data = model.sampleReal( N, M );
    SpectralHiddenMarkovModel algo = new SpectralHiddenMarkovModel(K, D);

    SimpleMatrix O_ = algo.run( data );

    O_ = MatrixOps.alignMatrix( O_, O, true );

    double err = MatrixOps.norm( O.minus( O_ ) );
    System.out.println( err );

    Assert.assertTrue( MatrixOps.allclose( O, O_, 1e-1 ) );
  }

  @Test
  public void testSmall() throws NumericalException {
    int N = (int) 1e6;
    int M = (int) 6;

    int K = 2;
    int E = 3;
    int D = 3;

    GenerationOptions genOptions = new GenerationOptions(K, E);
    FeatureOptions featureOptions = new FeatureOptions(D, "eye");

    RealHiddenMarkovModel model = RealHiddenMarkovModel.generate( genOptions, featureOptions );

    testHiddenMarkovModel( N, M, model );
  }
  @Test
  public void testSmallWithNoise() throws NumericalException {
    int N = (int) 1e6;
    int M = (int) 6;

    int K = 2;
    int E = 3;
    int D = 3;
    double noise = 1e-1;

    GenerationOptions genOptions = new GenerationOptions(K, E);
    FeatureOptions featureOptions = new FeatureOptions(D, "eye", noise);

    RealHiddenMarkovModel model = RealHiddenMarkovModel.generate( genOptions, featureOptions );

    testHiddenMarkovModel( N, M, model );
  }

  @Test
  public void testSmallRandom() throws NumericalException {
    int N = (int) 1e6;
    int M = (int) 6;

    int K = 2;
    int E = 3;
    int D = 3;

    GenerationOptions genOptions = new GenerationOptions(K, E);
    FeatureOptions featureOptions = new FeatureOptions(D, "random");

    RealHiddenMarkovModel model = RealHiddenMarkovModel.generate( genOptions, featureOptions );

    testHiddenMarkovModel( N, M, model );
  }

  @Test
  public void testMedium() throws NumericalException {
    int N = (int) 1e6;
    int M = (int) 10;

    int K = 3;
    int E = 6;
    int D = 6;

    GenerationOptions genOptions = new GenerationOptions(K, E);
    FeatureOptions featureOptions = new FeatureOptions(D, "eye");

    RealHiddenMarkovModel model = RealHiddenMarkovModel.generate( genOptions, featureOptions );

    testHiddenMarkovModel( N, M, model );
  }

  @Test
  public void testMediumProjection() throws NumericalException {
    int N = (int) 1e6;
    int M = (int) 10;

    int K = 3;
    int E = 40;
    int D = 6;

    GenerationOptions genOptions = new GenerationOptions(K, E);
    FeatureOptions featureOptions = new FeatureOptions(D, "random");

    RealHiddenMarkovModel model = RealHiddenMarkovModel.generate( genOptions, featureOptions );

    testHiddenMarkovModel( N, M, model );
  }

  @Test
  public void testMediumProjectionWithNoise() throws NumericalException {
    int N = (int) 1e6;
    int M = (int) 10;

    int K = 3;
    int E = 40;
    int D = 6;
    double noise = 1e-1;

    GenerationOptions genOptions = new GenerationOptions(K, E);
    FeatureOptions featureOptions = new FeatureOptions(D, "random", noise);

    RealHiddenMarkovModel model = RealHiddenMarkovModel.generate( genOptions, featureOptions );

    testHiddenMarkovModel( N, M, model );
  }

}

