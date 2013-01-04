/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;

import learning.models.HiddenMarkovModel;
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
public class HiddenMarkovModelsTest {

  public void testModel( int N, GenerationOptions options ) {
    int K = (int) options.stateCount;
    int D = (int) options.emissionCount;

    HiddenMarkovModel model = HiddenMarkovModel.generate( options );
    Pair<int[], int[]> data = model.sampleWithHiddenVariables( N );
    double[] observed = MatrixFactory.castToDouble( data.getValue0() );
    double[] hidden = MatrixFactory.castToDouble( data.getValue1() );

    Assert.assertTrue( observed.length == N );
    Assert.assertTrue( MatrixOps.min( hidden ) >= 0 && MatrixOps.max( hidden ) < K );
    Assert.assertTrue( MatrixOps.min( observed ) >= 0 && MatrixOps.max( observed ) < D );

  }
  public void testModel( GenerationOptions options ) {
    testModel( 10, options );
  }

  @Test
  public void testDefault() {
    int K = 2;
    int D = 3;

    GenerationOptions options = new GenerationOptions(K, D);
    testModel( options );
  }

  @Test
  public void testLarge() {
    int K = 10;
    int D = 100;

    GenerationOptions options = new GenerationOptions(K, D);
    testModel( options );
  }

  public void testViterbi(HiddenMarkovModel model, int N) {
    Pair<int[], int[]> data = model.sampleWithHiddenVariables( N );
    int[] observed = data.getValue0(); int[] hidden = data.getValue1();

    int[] hidden_ = model.viterbi( observed );
    double actual_lhood = model.likelihood( observed, hidden );
    double map_lhood = model.likelihood( observed, hidden_ );

    Assert.assertTrue( actual_lhood <= map_lhood );
  }

  @Test
  public void testViterbi() {
    int K = 5;
    int D = 10;
    int N = 100;
    int M = 5;

    GenerationOptions options = new GenerationOptions(K, D);
    HiddenMarkovModel model = HiddenMarkovModel.generate( options );

    for( int n = 0; n < N; n++ ) {
      testViterbi( model, M );
    }
  }

  public void testForwardBackward(HiddenMarkovModel model, int N) {
    Pair<int[], int[]> data = model.sampleWithHiddenVariables( N );
    int[] observed = data.getValue0(); int[] hidden = data.getValue1();

    double[][] posterior = model.forwardBackward( observed );
    // TODO: Test the posteriori for some property.
  }


  @Test
  public void testForwardBackward() {
    int K = 3;
    int D = 5;
    int N = 10;
    int M = 20;

    GenerationOptions options = new GenerationOptions(K, D);
    HiddenMarkovModel model = HiddenMarkovModel.generate( options );

    for( int n = 0; n < N; n++ ) {
      testForwardBackward( model, M );
    }
  }


}

