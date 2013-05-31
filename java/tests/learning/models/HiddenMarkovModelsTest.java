/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import learning.linalg.*;

import learning.models.HiddenMarkovModel;
import learning.models.HiddenMarkovModel.Params;
import learning.models.HiddenMarkovModel.GenerationOptions;

import java.util.*;

import fig.basic.Fmt;
import fig.basic.LogInfo;

import org.javatuples.*;
import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

/**
 * Test code for various dimensions.
 */
public class HiddenMarkovModelsTest {
  // Params taken from
  // http://www.indiana.edu/~iulg/moss/hmmcalculations.pdf
  Params hmm1;
  Params hmm2;

  @Before
  public void setup() {
    hmm1 = new Params(2,2);

    hmm1.pi[0] = 0.85;
    hmm1.pi[1] = 0.15;

    hmm1.T[0][0] = 0.3;
    hmm1.T[0][1] = 0.7;
    hmm1.T[1][0] = 0.1;
    hmm1.T[1][1] = 0.9;

    hmm1.O[0][0] = 0.4;
    hmm1.O[0][1] = 0.6;
    hmm1.O[1][0] = 0.5;
    hmm1.O[1][1] = 0.5;


    hmm2 = new Params(2,2);

    hmm2.pi[0] = 0.20;
    hmm2.pi[1] = 0.80;

    hmm2.T[0][0] = 0.9;
    hmm2.T[0][1] = 0.1;
    hmm2.T[1][0] = 0.1;
    hmm2.T[1][1] = 0.9;

    hmm2.O[0][0] = 0.5;
    hmm2.O[0][1] = 0.5;
    hmm2.O[1][0] = 0.5;
    hmm2.O[1][1] = 0.5;
  }

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

  //@Test
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

  @Test
  public void testForward() {
    HiddenMarkovModel model = new HiddenMarkovModel(hmm1);

    // alpha
    int[] o1 = {0,1,1,0};
    double[][] f1 = {
          { 0.34, 0.075, },
          { 0.0657, 0.15275, },
          { 0.020991, 0.0917325, },
          { 0.00618822, 0.048626475 }
        };
    double[][] f1_ = model.forward(o1);

    Assert.assertTrue( MatrixOps.allclose( f1_, f1 ) );
    
    int[] o2 = {1,0,1};
    double[][] f2 = {
      { 0.51, 0.075, },
      { 0.0642, 0.21225, },
      { 0.024291, 0.1179825, }
    };

    double[][] f2_ = model.forward(o2);
    Assert.assertTrue( MatrixOps.allclose( f2_, f2 ) );
  }
  
  @Test
  public void testBackward() {
    HiddenMarkovModel model = new HiddenMarkovModel(hmm1);
      // alpha
    int[] o1 = {0,1,1,0};
    double[][] b1 = {
      {0.133143, 0.127281},
      {0.2561, 0.2487},
      {0.4700, 0.4900},
      {1.0, 1.0}};
    double[][] b1_ = model.backward(o1);
    Assert.assertTrue( MatrixOps.allclose( b1_, b1 ) );
    
    int[] o2 = {1,0,1};
    double[][] b2 = {
      {0.24210, 0.25070},
      {0.53, 0.51},
      {1, 1}};

    double[][] b2_ = model.backward(o2);
    Assert.assertTrue( MatrixOps.allclose( b2_, b2 ) );
  }

  //@Test
  // The posteriors don't seem to be quite correct.
  public void testBaumWelch() {
    HiddenMarkovModel model = new HiddenMarkovModel(hmm1);

    Params hmm2 = new Params(2,2);
    hmm2.pi[0] = 0.846;
    hmm2.pi[1] = 0.154;

    hmm2.T[0][0] = 0.298;
    hmm2.T[0][1] = 0.702;
    hmm2.T[1][0] = 0.106;
    hmm2.T[1][1] = 0.894;

    hmm2.O[0][0] = 0.357;
    hmm2.O[0][1] = 0.643;
    hmm2.O[1][0] = 0.4292;
    hmm2.O[1][1] = 0.5708;

    int[][] X = {
      {0,1,1,0}, // 1 : 2
      {1,0,1},
      {1,0,1}
    };
    model.baumWelchStep(X);

    MatrixOps.printVector( model.params.pi );
    MatrixOps.printArray( model.params.T );
    MatrixOps.printArray( model.params.O );

    Assert.assertTrue( 
        MatrixOps.allclose( model.params.pi, hmm2.pi ) );
    Assert.assertTrue( 
        MatrixOps.allclose( model.params.T, hmm2.T ) );
    Assert.assertTrue( 
        MatrixOps.allclose( model.params.O, hmm2.O ) );
  }

  @Test
  public void testBaumWelch2() {
    //HiddenMarkovModel model1 = new HiddenMarkovModel(hmm2);
    HiddenMarkovModel model1 = HiddenMarkovModel.generate(
        new GenerationOptions(2,2));
    //int[][] X = {
    //  { 1, 0 },
    //  { 1, 0 },
    //  { 1, 0 },
    //  { 1, 0 },
    //};
    int N = 1000; int L = 10;
    int[][] X = new int[N][L];
    for(int i = 0; i < N; i++) {
      X[i] = model1.sample(L);
    }
    
    // Learn!
		Params init = Params.uniformWithNoise( new Random(), 2, 2, 1.0 );
		//Params init = model1.params.clone();
    HiddenMarkovModel model2 = new HiddenMarkovModel(init);
    double lhood = Double.NEGATIVE_INFINITY;
    for( int i = 0; i < 1000; i++ ) {
      double lhood_ = model2.baumWelchStep( X );
      LogInfo.logs( "%f - %f = %f", lhood_, lhood, lhood_ - lhood );
      double diff = lhood_ - lhood;
      Assert.assertTrue( diff >= 0 );
      if( diff < 1e-4 ) break;
      lhood = lhood_;
    }

    MatrixOps.printVector( model2.params.pi );
    MatrixOps.printArray( model2.params.T );
    MatrixOps.printArray( model2.params.O );

    LogInfo.logs( "----" );
    MatrixOps.printVector( model1.params.pi );
    MatrixOps.printArray( model1.params.T );
    MatrixOps.printArray( model1.params.O );



    // Assert.assertTrue( 
    //     MatrixOps.allclose( model2.params.pi, hmm1.pi ) );
    // Assert.assertTrue( 
    //     MatrixOps.allclose( model2.params.T, hmm1.T ) );
    // Assert.assertTrue( 
    //     MatrixOps.allclose( model2.params.O, hmm1.O ) );
  }
}

