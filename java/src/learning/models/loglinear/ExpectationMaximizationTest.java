package learning.models.loglinear;

import learning.models.ExponentialFamilyModel;
import learning.models.Params;
import learning.utils.Counter;
import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

/**
 * Test various regimes for EM
 */
public class ExpectationMaximizationTest {
  Random rnd = new Random(42);

  public void testRecovery(ExponentialFamilyModel<Example> model, Params trueParams, Params initParams,
                           boolean testBetterThanTrue, boolean testDataEquals) {
    ExpectationMaximization solver = new ExpectationMaximization();

    solver.backtrack.tolerance = 1e-3;
    solver.mIters = 1;
    solver.iters = 500;
    solver.diagnosticMode = true;

    if(testBetterThanTrue)
      solver.thetaRegularization = 0.0;
    else
      solver.thetaRegularization = 0.;//1e-2;
    // Generate examples from the model
    Counter<Example> data =  model.drawSamples(trueParams, rnd, (int) 1e5);

    double trueDataLhood = model.getLogLikelihood(trueParams, data);
    double estDataLhood_ = model.getLogLikelihood(initParams, data);

    Params finalParams = solver.solveEM(model, data, initParams);

    double estDataLhood = model.getLogLikelihood(finalParams, data);

    Assert.assertTrue( (estDataLhood - estDataLhood_) > -1e-2 );
    if(testBetterThanTrue)
      Assert.assertTrue( (estDataLhood - trueDataLhood) > -1e-2 ); // since we started there

    if(testDataEquals) {
      // Check that their marginals match
      Counter<Example> dist = model.getDistribution(trueParams);
      Counter<Example> dist_ = model.getDistribution(finalParams);

      for( Example ex : dist ) {
        Assert.assertEquals(dist.getCount(ex), dist_.getCount(ex), 1e-2);
      }
    }
  }

  @Test
  public void testMixtureModelEstimation1() {
    int K = 3; int D = 3; int L = 3;
    Models.MixtureModel modelA = new Models.MixtureModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    {
      double[] params = trueParams.toArray();
      for(int i = 0; i < params.length; i++)
        params[i] = Math.sin(i);
    }
    Params initParams = trueParams.copy();

    testRecovery(modelA, trueParams, initParams, true, true);
  }
  @Test
  public void testMixtureModelEstimation2() {
    int K = 3; int D = 3; int L = 3;
    Models.MixtureModel modelA = new Models.MixtureModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    {
      double[] params = trueParams.toArray();
      for(int i = 0; i < params.length; i++)
        params[i] = Math.sin(i);
    }
    Params initParams = trueParams.newParams();
    initParams.initRandom(rnd, 1.0);

    testRecovery(modelA, trueParams, initParams, false, true);
  }
  @Test
  public void testMixtureModelEstimation3() {
    int K = 3; int D = 3; int L = 3;
    Models.MixtureModel modelA = new Models.MixtureModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    trueParams.initRandom(rnd, 1.0);
    Params initParams = trueParams.copy();

    testRecovery(modelA, trueParams, initParams, true, true);
  }
  @Test
  public void testMixtureModelEstimation4() {
    int K = 3; int D = 3; int L = 3;
    Models.MixtureModel modelA = new Models.MixtureModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    trueParams.initRandom(rnd, 1.0);
    Params initParams = trueParams.newParams();
    initParams.initRandom(rnd, 1.0);

    testRecovery(modelA, trueParams, initParams, false, true);
  }

  @Test
  public void testHiddenMarkovModelEstimation1() {
    int K = 3; int D = 3; int L = 3;
    Models.HiddenMarkovModel modelA = new Models.HiddenMarkovModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    {
      double[] params = trueParams.toArray();
      for(int i = 0; i < params.length; i++)
        params[i] = Math.sin(i);
    }
    Params initParams = trueParams.copy();

    testRecovery(modelA, trueParams, initParams, true, true);
  }
  @Test
  public void testHiddenMarkovModelEstimation2() {
    int K = 3; int D = 3; int L = 3;
    Models.HiddenMarkovModel modelA = new Models.HiddenMarkovModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    {
      double[] params = trueParams.toArray();
      for(int i = 0; i < params.length; i++)
        params[i] = Math.sin(i);
    }
    Params initParams = trueParams.newParams();
    initParams.initRandom(rnd, 1.0);

    testRecovery(modelA, trueParams, initParams, false, true);
  }
  @Test
  public void testHiddenMarkovModelEstimation3() {
    int K = 3; int D = 3; int L = 3;
    Models.HiddenMarkovModel modelA = new Models.HiddenMarkovModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    trueParams.initRandom(rnd, 1.0);
    Params initParams = trueParams.copy();

    testRecovery(modelA, trueParams, initParams, true, true);
  }
  @Test
  public void testHiddenMarkovModelEstimation4() {
    int K = 3; int D = 3; int L = 3;
    Models.HiddenMarkovModel modelA = new Models.HiddenMarkovModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    trueParams.initRandom(rnd, 1.0);
    Params initParams = trueParams.newParams();
    initParams.initRandom(rnd, 1.0);

    testRecovery(modelA, trueParams, initParams, false, true);
  }

  @Test
  public void testUndirectedHiddenMarkovModelEstimation1() {
    int K = 3; int D = 3; int L = 3;
    UndirectedHiddenMarkovModel modelA = new UndirectedHiddenMarkovModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    {
      double[] params = trueParams.toArray();
      for(int i = 0; i < params.length; i++)
        params[i] = Math.sin(i);
    }
    Params initParams = trueParams.copy();

    testRecovery(modelA, trueParams, initParams, true, true);
  }
  @Test
  public void testUndirectedHiddenMarkovModelEstimation2() {
    int K = 3; int D = 3; int L = 3;
    UndirectedHiddenMarkovModel modelA = new UndirectedHiddenMarkovModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    {
      double[] params = trueParams.toArray();
      for(int i = 0; i < params.length; i++)
        params[i] = Math.sin(i);
    }
    Params initParams = trueParams.newParams();
    initParams.initRandom(rnd, 1.0);

    testRecovery(modelA, trueParams, initParams, false, true);
  }
  @Test
  public void testUndirectedHiddenMarkovModelEstimation3() {
    int K = 3; int D = 3; int L = 3;
    UndirectedHiddenMarkovModel modelA = new UndirectedHiddenMarkovModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    trueParams.initRandom(rnd, 1.0);
    Params initParams = trueParams.copy();

    testRecovery(modelA, trueParams, initParams, true, true);
  }
  @Test
  public void testUndirectedHiddenMarkovModelEstimation4() {
    int K = 3; int D = 3; int L = 3;
    UndirectedHiddenMarkovModel modelA = new UndirectedHiddenMarkovModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    trueParams.initRandom(rnd, 1.0);
    Params initParams = trueParams.newParams();
    initParams.initRandom(rnd, 1.0);

    testRecovery(modelA, trueParams, initParams, false, true);
  }

  @Test
  public void testUndirectedHiddenMarkovModelEstimationLarge() {
    int K = 5; int D = 100; int L = 3;
    UndirectedHiddenMarkovModel modelA = new UndirectedHiddenMarkovModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    trueParams.initRandom(rnd, 1.0);
    Params initParams = trueParams.newParams();
    initParams.initRandom(rnd, 1.0);

    testRecovery(modelA, trueParams, initParams, false, true);
  }
  @Test
  public void testUndirectedHiddenMarkovModelEstimationLarge1e3() {
    int K = 5; int D = 1000; int L = 3;
    UndirectedHiddenMarkovModel modelA = new UndirectedHiddenMarkovModel(K, D, L);
    // Set parameters to be sin
    Params trueParams = modelA.newParams();
    trueParams.initRandom(rnd, 1.0);
    Params initParams = trueParams.newParams();
    initParams.initRandom(rnd, 1.0);

    testRecovery(modelA, trueParams, initParams, false, true);
  }

}
