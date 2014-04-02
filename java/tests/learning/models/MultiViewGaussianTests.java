package learning.models;

import fig.basic.Fmt;
import fig.basic.LogInfo;
import junit.framework.Assert;
import learning.common.Counter;
import learning.linalg.MatrixOps;
import org.junit.Test;

import java.util.Random;

/**
 * Test cases to ensure that the multi-view Gaussian is correctly implemented
 */
public class MultiViewGaussianTests {

  Random rnd = new Random(42);

  @Test
  public void testGenerate() {
    MultiViewGaussian model; MultiViewGaussian.Parameters params;
    model = new MultiViewGaussian(2,2,3);
    params = model.newParams();
    params.initRandom(rnd, 1.0);
    Assert.assertTrue(params.isValid());
    Assert.assertEquals(model.getLogLikelihood(params), 0.0, 1e-5);

    model = new MultiViewGaussian(2,3,3);
    params = model.newParams();
    params.initRandom(rnd, 1.0);
    Assert.assertTrue(params.isValid());
    Assert.assertEquals(model.getLogLikelihood(params), 0.0, 1e-5);

    model = new MultiViewGaussian(3,2,3);
    params = model.newParams();
    params.initRandom(rnd, 1.0);
    Assert.assertTrue(params.isValid());
    Assert.assertEquals(model.getLogLikelihood(params), 0.0, 1e-5);
  }

  @Test
  public void testExactMarginals() {
    MultiViewGaussian model; MultiViewGaussian.Parameters params;
    model = new MultiViewGaussian(2,2,3);
    params = model.newParams();
    params.initRandom(rnd, 1.0);

    MultiViewGaussian.Parameters params_= (MultiViewGaussian.Parameters) model.getMarginals(params);
    Assert.assertTrue(MatrixOps.allclose(params.toArray(), params_.toArray()));
  }

  @Test
  public void testEmpiricalMarginals() {
    MultiViewGaussian model; MultiViewGaussian.Parameters params;
    model = new MultiViewGaussian(2,2,3);
    params = model.newParams();
    params.initRandom(rnd, 1.0);

    Counter<double[][]> data = model.drawSamples(params, new Random(10), (int) 1e2);

    MultiViewGaussian.Parameters params_= (MultiViewGaussian.Parameters) model.getMarginals(params, data);
    LogInfo.log(Fmt.D(MatrixOps.diff(params.toArray(), params_.toArray())));
    Assert.assertTrue(MatrixOps.allclose(params.toArray(), params_.toArray(), 1e-5));
  }


}
