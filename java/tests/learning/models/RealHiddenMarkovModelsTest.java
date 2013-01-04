/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;

import learning.models.RealHiddenMarkovModel;
import learning.models.RealHiddenMarkovModel.FeatureOptions;
import learning.models.HiddenMarkovModel.GenerationOptions;

import java.util.Arrays;

import org.javatuples.*;
import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

/**
 * Test code for various dimensions.
 */
public class RealHiddenMarkovModelsTest {

  public void testModel( int N, int M, GenerationOptions options, FeatureOptions featureOptions ) {
    int K = (int) options.stateCount;
    int E = (int) options.emissionCount;
    int D = (int) featureOptions.dimension;

    RealHiddenMarkovModel model = RealHiddenMarkovModel.generate( options, featureOptions );
    Pair<double[][][], int[][]> data = model.sampleRealWithHiddenVariables( N, M );
    double[][][] observed = data.getValue0();

    Assert.assertTrue( observed.length == N );
    Assert.assertTrue( observed[0].length == M );
    Assert.assertTrue( observed[0][0].length == D );
    for( int i = 0; i < N; i++ ) {
      double[] hidden = MatrixFactory.castToDouble( data.getValue1()[0] );
      Assert.assertTrue( MatrixOps.min( hidden ) >= 0 && MatrixOps.max( hidden ) < K );
    }
  }
  public void testModel( GenerationOptions options, FeatureOptions featureOptions ) {
    testModel( 20, 10, options, featureOptions );
  }

  @Test
  public void testDefault() {
    int K = 2;
    int E = 3;

    GenerationOptions options = new GenerationOptions(K, E);
    FeatureOptions featureOptions = new FeatureOptions( E, "eye" );
    testModel( options, featureOptions );
  }

  @Test
  public void testDefaultWithRandomFeatures() {
    int K = 2;
    int E = 10;

    GenerationOptions options = new GenerationOptions(K, E);
    FeatureOptions featureOptions = new FeatureOptions( E, "random" );

    testModel( options, featureOptions );
  }

  @Test
  public void testLarge() {
    int K = 10;
    int E = 100;

    GenerationOptions options = new GenerationOptions(K, E);
    FeatureOptions featureOptions = new FeatureOptions( E, "eye" );
    testModel( options, featureOptions );
  }

  @Test
  public void testLargeWithRandomFeatures() {
    int K = 10;
    int E = 100;
    int D = 10;

    GenerationOptions options = new GenerationOptions(K, E);
    FeatureOptions featureOptions = new FeatureOptions( D, "random" );
    testModel( options, featureOptions );
  }


}

