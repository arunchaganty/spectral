/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import fig.basic.Pair;
import learning.common.Counter;
import learning.linalg.*;

import java.util.*;

import fig.basic.LogInfo;

import learning.models.loglinear.Example;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

import static fig.basic.LogInfo.log;
import static learning.common.Utils.outputList;
import static learning.models.HiddenMarkovModel.*;

/**
 * Test code for various dimensions.
 */
public class HiddenMarkovModelsTest {
  // Params taken from
  // http://www.indiana.edu/~iulg/moss/hmmcalculations.pdf
  HiddenMarkovModel model;
  HiddenMarkovModel.Parameters hmm1;
  HiddenMarkovModel.Parameters hmm2;

  Random testRandom = new Random(42);

  @Before
  public void setup() {
    model = new HiddenMarkovModel(2,2,3);
    hmm1 = model.newParams();

    hmm1.set(piFeature(0), 0.85);
    hmm1.set(piFeature(1), 0.15);

    hmm1.set(tFeature(0,0), 0.3);
    hmm1.set(tFeature(0,1), 0.7);
    hmm1.set(tFeature(1,0), 0.1);
    hmm1.set(tFeature(1,1), 0.9);

    hmm1.set(oFeature(0,0), 0.4);
    hmm1.set(oFeature(0,1), 0.6);
    hmm1.set(oFeature(1,0), 0.5);
    hmm1.set(oFeature(1,1), 0.5);

    hmm2 = model.newParams();

    hmm2.set(piFeature(0), 0.20 );
    hmm2.set(piFeature(1), 0.80 );

    hmm2.set(tFeature(0, 0), 0.9);
    hmm2.set(tFeature(0, 1), 0.1);
    hmm2.set(tFeature(1, 0), 0.1);
    hmm2.set(tFeature(1, 1), 0.9);

    hmm2.set(oFeature(0,0), 0.5);
    hmm2.set(oFeature(0,1), 0.5);
    hmm2.set(oFeature(1,0), 0.5);
    hmm2.set(oFeature(1,1), 0.5);
  }

  public void testModel( int K, int D, int L) {
    HiddenMarkovModel model = new HiddenMarkovModel(K,D,L);
    HiddenMarkovModel.Parameters params = model.newParams();
    params.initRandom(testRandom, 1.0);
    Example ex = model.drawSample(params, testRandom);

    Assert.assertTrue( ex.x.length == L );
    Assert.assertTrue( MatrixOps.min( ex.h ) >= 0 && MatrixOps.max( ex.h ) < K );
    Assert.assertTrue( MatrixOps.min( ex.x ) >= 0 && MatrixOps.max( ex.x ) < D );
  }

  @Test
  public void testDefault() {
    int K = 2;
    int D = 3;
    int L = 10;

    testModel( K, D, L);
  }

  @Test
  public void testLarge() {
    int K = 10;
    int D = 100;
    int L = 10;

    testModel( K, D, L);
  }

  public void testViterbi(HiddenMarkovModel model) {
    HiddenMarkovModel.Parameters params = model.newParams();
    params.initRandom(testRandom, 1.0);
    Example ex = model.drawSample(params, testRandom);

    int[] h = ex.h;
    int[] h_ = model.viterbi(params, ex);
    double true_lhood = model.getFullProbability(params, ex);
    double map_lhood = model.getFullProbability(params, ex.withH(h_));

    Assert.assertTrue( true_lhood <= map_lhood );
  }

  @Test
  public void testViterbi() {
    int K = 5;
    int D = 10;
    int N = 100;
    int M = 5;

    HiddenMarkovModel model = new HiddenMarkovModel(K,D,M);

    for( int n = 0; n < N; n++ ) {
      testViterbi( model );
    }
  }

  @Test
  public void testForward() {
    double[][] f0 = {
            { 0.85, 0.15 },
            { 0.27, 0.73 },
            { 0.154, 0.846 } };
    Pair<Double, double[][]> f0c_ = model.forward(hmm1, null);
    double[][] f0_ = f0c_.getSecond();

    Assert.assertTrue( MatrixOps.allclose( f0_, f0 ) );

    // alpha
    int[] o1 = {0,1,1,0};
    Example ex1 = new Example(o1);
    double[][] f1 = {
      { 0.81928, 0.18072 },
      { 0.30076, 0.69924 },
      { 0.18622, 0.81378 },
      { 0.11289, 0.88711 }, };
    Pair<Double, double[][]> f1c_ = model.forward(hmm1, ex1);
    double[][] f1_ = f1c_.getSecond();

    Assert.assertTrue( MatrixOps.allclose( f1_, f1 ) );
    
    int[] o2 = {1,0,1};
    Example ex2 = new Example(o2);
    double[][] f2 = {
      { 0.2, 0.8 },
      { 0.26, 0.74 },
      { 0.308, 0.692 },
    };

    Pair<Double, double[][]> f2c_ = model.forward(hmm2, ex2);
    double[][] f2_ = f2c_.getSecond();
    Assert.assertTrue( MatrixOps.allclose( f2_, f2 ) );
  }
  
  @Test
  public void testBackward() {
    double[][] b0 = {
            { 0.5, 0.5 },
            { 0.5, 0.5 },
            { 0.5, 0.5 },
    };
    double[][] b0_ = model.backward(hmm1, null);
    Assert.assertTrue( MatrixOps.allclose( b0_, b0 ) );

      // alpha
    int[] o1 = {0,1,1,0};
    Example ex1 = new Example(o1);
    double[][] b1 = {
      { 0.51125, 0.48875 },
      { 0.50733, 0.49267 },
      { 0.48958, 0.51042 },
      { 0.50000, 0.50000 },
    };

    double[][] b1_ = model.backward(hmm1, ex1);
    Assert.assertTrue( MatrixOps.allclose( b1_, b1 ) );
    
    int[] o2 = {1,0,1};
    Example ex2 = new Example(o2);
    double[][] b2 = {
      { 0.49127, 0.50873 },
      { 0.50962, 0.49038 },
      { 0.50000, 0.50000 },
    };

    double[][] b2_ = model.backward(hmm1, ex2);
    Assert.assertTrue( MatrixOps.allclose( b2_, b2 ) );

    // - hmm2 is stupid in that all the observations are 0.
    double[][] b3 = {
            { 0.50000, 0.50000 },
            { 0.50000, 0.50000 },
            { 0.50000, 0.50000 },
    };
    double[][] b3_ = model.backward(hmm2, ex2);
    Assert.assertTrue( MatrixOps.allclose( b3_, b3 ) );

  }

  @Test
  public void testMarginals() {
    HiddenMarkovModel.Parameters marginals;

    Example ex1 = new Example(new int[] {1,0,1}) ;
    marginals = (HiddenMarkovModel.Parameters) model.getMarginals(hmm1, ex1);
    Assert.assertTrue(marginals.isValid());

    marginals = (HiddenMarkovModel.Parameters) model.getMarginals(hmm1);
    Assert.assertTrue(marginals.isValid());
    // By definition
    for(int i = 0; i < marginals.weights.length; i++) {
      Assert.assertTrue(Math.abs(marginals.weights[i] - hmm1.weights[i]) < 1e-1);
    }

    Counter<Example> data = model.getDistribution(hmm1);
    marginals = (HiddenMarkovModel.Parameters) model.getMarginals(hmm1, data);
    Assert.assertTrue(marginals.isValid());
    // By definition
    for(int i = 0; i < marginals.weights.length; i++) {
      Assert.assertTrue(Math.abs(marginals.weights[i] - hmm1.weights[i]) < 1e-1);
    }

  }

  @Test
  public void testSampleMarginalsExact() {
    Counter<Example> data = model.getDistribution(hmm1);
    HiddenMarkovModel.Parameters marginal = model.getSampleMarginals(data);
    for(int i = 0; i < marginal.weights.length; i++) {
      Assert.assertTrue(Math.abs(marginal.weights[i] - hmm1.weights[i]) < 1e-1);
    }
  }


  @Test
  public void testSampleMarginals() {
    Counter<Example> data = model.drawSamples(hmm1, new Random(1), 1000000);
    HiddenMarkovModel.Parameters marginal = model.getSampleMarginals(data);
    for(int i = 0; i < marginal.weights.length; i++) {
      Assert.assertTrue(Math.abs(marginal.weights[i] - hmm1.weights[i]) < 1e-1);
    }
  }

  @Test
  // The posteriors don't seem to be quite correct.
  public void testBaumWelch() {
    Params hmm3 = model.newParams();

    hmm3.set(piFeature(0), 0.846 );
    hmm3.set(piFeature(1), 0.154 );

    hmm3.set(tFeature(0, 0), 0.298);
    hmm3.set(tFeature(0, 1), 0.702);
    hmm3.set(tFeature(1, 0), 0.106);
    hmm3.set(tFeature(1, 1), 0.894);

    hmm3.set(oFeature(0,0), 0.357 );
    hmm3.set(oFeature(0,1), 0.643 );
    hmm3.set(oFeature(1,0), 0.4292);
    hmm3.set(oFeature(1,1), 0.5708);

    Counter<Example> data = new Counter<>();
    data.add(new Example(new int[]{0,1,1,0}));
    data.add(new Example(new int[]{1,0,1}));
    data.add(new Example(new int[]{1,0,1}));

    // Simple EM
    Params marginals = model.newParams();
    model.updateMarginals(hmm3, data, 1.0, marginals);
//    model.baumWelchStep(X);

    log(marginals);

//    Assert.assertTrue(
//        MatrixOps.allclose( model.params.pi, hmm3.pi ) );
//    Assert.assertTrue(
//        MatrixOps.allclose( model.params.T, hmm3.T ) );
//    Assert.assertTrue(
//        MatrixOps.allclose( model.params.O, hmm3.O ) );
  }

  @Test
  public void testEM() {
    Params hmm3 = model.newParams();

    hmm3.set(piFeature(0), 0.846 );
    hmm3.set(piFeature(1), 0.154 );

    hmm3.set(tFeature(0, 0), 0.298);
    hmm3.set(tFeature(0, 1), 0.702);
    hmm3.set(tFeature(1, 0), 0.106);
    hmm3.set(tFeature(1, 1), 0.894);

    hmm3.set(oFeature(0,0), 0.357 );
    hmm3.set(oFeature(0,1), 0.643 );
    hmm3.set(oFeature(1,0), 0.4292);
    hmm3.set(oFeature(1,1), 0.5708);

//    Counter<Example> data = model.drawSamples(hmm1, testRandom, (int) 1e6);
    Counter<Example> data = model.getDistribution(hmm1);
    Params marginals = model.newParams();

    Params params = model.newParams();
    params.initRandom(testRandom, 1.0);
    double oldLhood = Double.NEGATIVE_INFINITY;
    for(int i = 0; i < 1000; i++) {
      // Simple EM
      marginals.clear();
      model.updateMarginals(params, data, 1.0, marginals);
      double diff = params.computeDiff(marginals, null);
      params.copyOver(marginals);

      double lhood = model.getLogLikelihood(params,data);

      log(outputList(
              "iter", i,
              "likelihood", lhood,
              "diff", diff
      ));
      Assert.assertTrue( lhood - oldLhood > -1e-2); // Numerical error.
      oldLhood = lhood;

      if( diff < 1e-3 ) break;
    }

    log("True likelihood: " + model.getLogLikelihood(hmm1, data));
    log("Fit likelihood: " + model.getLogLikelihood(params, data));

    log(marginals);

//    Assert.assertTrue(
//        MatrixOps.allclose( model.params.pi, hmm3.pi ) );
//    Assert.assertTrue(
//        MatrixOps.allclose( model.params.T, hmm3.T ) );
//    Assert.assertTrue(
//        MatrixOps.allclose( model.params.O, hmm3.O ) );
  }

//  @Test
//  public void testBaumWelch2() {
//    //HiddenMarkovModel model1 = new HiddenMarkovModel(hmm2);
//    HiddenMarkovModel model1 = generate(
//            new GenerationOptions(2, 2));
//    //int[][] X = {
//    //  { 1, 0 },
//    //  { 1, 0 },
//    //  { 1, 0 },
//    //  { 1, 0 },
//    //};
//    int N = 100; int L = 10;
//    int[][] X = new int[N][L];
//    for(int i = 0; i < N; i++) {
//      X[i] = model1.sample(L);
//    }
//
//    // Learn!
//		Params init = Params.uniformWithNoise( new Random(), 2, 2, 1.0 );
//		//Params init = model1.params.clone();
//    HiddenMarkovModel model2 = new HiddenMarkovModel(init);
//    double lhood = Double.NEGATIVE_INFINITY;
//    for( int i = 0; i < 1000; i++ ) {
//      double lhood_ = model2.baumWelchStep( X );
//      LogInfo.logs( "%f - %f = %f", lhood_, lhood, lhood_ - lhood );
//      double diff = lhood_ - lhood;
//      Assert.assertTrue( diff >= -1e-4 ); // Shouldn't be too small
//      if( Math.abs(diff) < 1e-4 ) break;
//      lhood = lhood_;
//    }
//
//    MatrixOps.printVector( model2.params.pi );
//    MatrixOps.printArray( model2.params.T );
//    MatrixOps.printArray( model2.params.O );
//    LogInfo.logs( "lhood: " + model2.compute(X,null));
//
//    LogInfo.logs( "----" );
//    MatrixOps.printVector( model1.params.pi );
//    MatrixOps.printArray( model1.params.T );
//    MatrixOps.printArray( model1.params.O );
//    LogInfo.logs( "lhood: " + model1.compute(X,null));
//
//    double diff = model1.toParams().computeDiff(model2.toParams(),null);
//    LogInfo.log( "diff: " + diff);
//    Assert.assertTrue( diff < 1e-2);
//  }
}

