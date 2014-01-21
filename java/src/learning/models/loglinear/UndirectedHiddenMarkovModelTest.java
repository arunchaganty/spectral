package learning.models.loglinear;

import fig.basic.Pair;
import learning.models.loglinear.UndirectedHiddenMarkovModel.Parameters;
import learning.common.Counter;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

/**
 *
 */
public class UndirectedHiddenMarkovModelTest {
  UndirectedHiddenMarkovModel model;
  Parameters model0;
  Parameters model1;

  @Before
  public void initialize() {
    int K = 2, D = 2, L = 3;
    model = new UndirectedHiddenMarkovModel(K, D, L);
    model0 = (Parameters) model.newParams();

    model1 = (Parameters) model.newParams();
    model1.weights[model.o(0,0)] = Math.log(1.);
    model1.weights[model.o(0,1)] = Math.log(2.);
    model1.weights[model.o(1,0)] = Math.log(1.);
    model1.weights[model.o(1,1)] = Math.log(1.);

    model1.weights[model.t(0,0)] = Math.log(1.);
    model1.weights[model.t(0,1)] = Math.log(2.);
    model1.weights[model.t(1,0)] = Math.log(3.);
    model1.weights[model.t(1,1)] = Math.log(4.);
  }

  @Test
  public void testForward() {
    int T = model.L; int K = model.K; int D = model.D;
    // The first model, everything has unit weights.
    {
      Pair<Double, double[][]> Zforwards = model.forward(model0, null);
      double Z = Zforwards.getFirst(); double[][] forwards = Zforwards.getSecond();
      Assert.assertEquals(Math.log(K * D) * T, Z, 1e-4);
      for( int t = 0; t < T; t++ ) {
        for( int y = 0; y < K; y++ ) {
          Assert.assertEquals((1. / K), forwards[t][y], 1e-4);
        }
      }
    }
    //

    {
      Example ex = new Example(new int[]{0,0,0}, new int[]{0,0,0});
      T = ex.x.length;
      Pair<Double, double[][]> Zforwards = model.forward(model0, ex);
      double Z = Zforwards.getFirst(); double[][] forwards = Zforwards.getSecond();
      Assert.assertEquals(Math.log(K) * T, Z, 1e-4);
      for( int t = 0; t < T; t++ ) {
        for( int y = 0; y < K; y++ ) {
          Assert.assertEquals(1. / K, forwards[t][y], 1e-4);
        }
      }
    }

    // The second model, things have non-unit weights.
    {
      Pair<Double, double[][]> Zforwards = model.forward(model1, null);
      double Z = Zforwards.getFirst(); double[][] forwards = Zforwards.getSecond();

      Assert.assertEquals(6.4997, Z, 1e-3);
      Assert.assertArrayEquals(forwards[0], new double[]{(0.6), (0.4)}, 1e-2);
      Assert.assertArrayEquals(forwards[1], new double[]{(0.49), (0.51)}, 1e-2);
      Assert.assertArrayEquals(forwards[2], new double[]{(0.501), (0.499)}, 1e-2);
    }
    //

    {
      Example ex = new Example(new int[]{0,0,0}, new int[]{0,0,0});
      T = ex.x.length;
      Pair<Double, double[][]> Zforwards = model.forward(model1, ex);
      double Z = Zforwards.getFirst(); double[][] forwards = Zforwards.getSecond();
      Assert.assertEquals(Math.log(54), Z, 1e-4);
      Assert.assertArrayEquals(forwards[0], new double[]{(0.5),(0.5)}, 1e-2);
      Assert.assertArrayEquals(forwards[1], new double[]{(.4), (0.6)}, 1e-2);
      Assert.assertArrayEquals(forwards[2], new double[]{(0.407), (0.593) }, 1e-2);
    }
  }

  @Test
  public void testBackward() {
    int T = model.L; int K = model.K; int D = model.D;
    // The first model, everything has unit weights.
    {
      double[][] backwards = model.backward(model0, null);
      for( int t = 0; t < T; t++ ) {
        for( int y = 0; y < K; y++ ) {
          Assert.assertEquals((1. / K), backwards[t][y], 1e-4);
        }
      }
    }
    //

    {
      Example ex = new Example(new int[]{0,0,0}, new int[]{0,0,0});
      T = ex.x.length;
      double[][] backwards = model.backward(model0, ex);
      for( int t = 0; t < T; t++ ) {
        for( int y = 0; y < K; y++ ) {
          Assert.assertEquals((1. / K), backwards[t][y], 1e-4);
        }
      }
    }

    // The second model, things have non-unit weights.
    {
      double[][] backwards = model.backward(model1, null);
      Assert.assertArrayEquals(new double[]{(0.5),   (0.5)}, backwards[2], 1e-2);
      Assert.assertArrayEquals(new double[]{(0.292), (0.708)}, backwards[1], 1e-2);
      Assert.assertArrayEquals(new double[]{(0.309), (0.691)}, backwards[0], 1e-2);
    }
    //

    {
      Example ex = new Example(new int[]{0,0,0}, new int[]{0,0,0});
      T = ex.x.length;
      double[][] backwards = model.backward(model1, ex);
      Assert.assertArrayEquals(new double[]{(0.5),   (0.5)}, backwards[2], 1e-2);
      Assert.assertArrayEquals(new double[]{(0.3),   (0.7)}, backwards[1], 1e-2);
      Assert.assertArrayEquals(new double[]{(0.314), (0.686) }, backwards[0], 1e-2);
    }
  }

  @Test
  public void testEdgeMarginals() {
    int T = model.L; int K = model.K; int D = model.D;
    // The first model, everything has unit weights.
    {
      double[][][] marginals = model.computeEdgeMarginals(model0, null);
      Assert.assertArrayEquals(new double[] {0.5, 0.5}, marginals[0][0], 1e-2);
      Assert.assertArrayEquals(new double[] {0.0, 0.0}, marginals[0][1], 1e-2);
      Assert.assertArrayEquals(new double[] {0.25, 0.25}, marginals[1][0], 1e-2);
      Assert.assertArrayEquals(new double[] {0.25, 0.25}, marginals[1][1], 1e-2);
      Assert.assertArrayEquals(new double[] {0.25, 0.25}, marginals[2][0], 1e-2);
      Assert.assertArrayEquals(new double[] {0.25, 0.25}, marginals[2][1], 1e-2);
    }
    //

    {
      Example ex = new Example(new int[]{0,0,0}, new int[]{0,0,0});
      T = ex.x.length;
      double[][][] marginals = model.computeEdgeMarginals(model0, ex);
      Assert.assertArrayEquals(new double[] {0.5, 0.5}, marginals[0][0], 1e-2);
      Assert.assertArrayEquals(new double[] {0.0, 0.0}, marginals[0][1], 1e-2);
      Assert.assertArrayEquals(new double[] {0.25, 0.25}, marginals[1][0], 1e-2);
      Assert.assertArrayEquals(new double[] {0.25, 0.25}, marginals[1][1], 1e-2);
      Assert.assertArrayEquals(new double[] {0.25, 0.25}, marginals[2][0], 1e-2);
      Assert.assertArrayEquals(new double[] {0.25, 0.25}, marginals[2][1], 1e-2);
    }

    // The second model, things have non-unit weights.
    {
      double[][][] marginals = model.computeEdgeMarginals(model1, null);
      // TODO: Write asserts
//      Assert.assertArrayEquals(new double[]{0.5, 0.5}, backwards[2], 1e-2);
//      Assert.assertArrayEquals(new double[]{0.292, 0.708}, backwards[1], 1e-2);
//      Assert.assertArrayEquals(new double[]{0.309, 0.691 }, backwards[0], 1e-2);
    }
    //

    {
      Example ex = new Example(new int[]{0,0,0}, new int[]{0,0,0});
      T = ex.x.length;
      double[][][] marginals = model.computeEdgeMarginals(model1, ex);
      // TODO: Write asserts
//      Assert.assertArrayEquals(new double[]{0.5, 0.5}, backwards[2], 1e-2);
//      Assert.assertArrayEquals(new double[]{0.3, 0.7}, backwards[1], 1e-2);
//      Assert.assertArrayEquals(new double[]{0.314, 0.686 }, backwards[0], 1e-2);
    }
  }

  @Test
  public void testLikelihood() {
    Example ex = new Example(new int[]{0,0,0}, new int[]{0,0,0});
    Assert.assertEquals(Math.log(2 * 2) * 3, model.getLogLikelihood(model0) , 1e-4);
    Assert.assertEquals(Math.log(2) * 3, model.getLogLikelihood(model0, ex) , 1e-4);
    Assert.assertEquals(Math.log(665), model.getLogLikelihood(model1) , 1e-4);
    Assert.assertEquals(Math.log(54), model.getLogLikelihood(model1, ex) , 1e-4);
  }

  @Test
  public void testMarginals() {
    Example ex = new Example(new int[]{0,0,0}, new int[]{0,0,0});
    {
      Parameters marginals = (Parameters) model.getMarginals(model0);
      int T = model.L;
      Assert.assertEquals(marginals.weights[model.o(0, 0)], 0.25 * T, 1e-3 );
      Assert.assertEquals(marginals.weights[model.o(0, 1)], 0.25 * T, 1e-3 );
      Assert.assertEquals(marginals.weights[model.o(1, 0)], 0.25 * T, 1e-3 );
      Assert.assertEquals(marginals.weights[model.o(1, 1)], 0.25 * T, 1e-3 );

      Assert.assertEquals(marginals.weights[model.t(0, 0)], 0.25 * (T-1), 1e-3);
      Assert.assertEquals(marginals.weights[model.t(0, 1)], 0.25 * (T-1), 1e-3);
      Assert.assertEquals(marginals.weights[model.t(1, 0)], 0.25 * (T-1), 1e-3);
      Assert.assertEquals(marginals.weights[model.t(1, 1)], 0.25 * (T-1), 1e-3);
    }
    {
      int T = model.L;
      Parameters marginals = (Parameters) model.getMarginals(model0, ex);
      Assert.assertEquals(marginals.weights[model.o(0, 0)], 0.5 * T, 1e-3 );
      Assert.assertEquals(marginals.weights[model.o(0, 1)], 0.0 * T, 1e-3 );
      Assert.assertEquals(marginals.weights[model.o(1, 0)], 0.5 * T, 1e-3 );
      Assert.assertEquals(marginals.weights[model.o(1, 1)], 0.0 * T, 1e-3 );

      Assert.assertEquals(marginals.weights[model.t(0, 0)], 0.25 * (T-1), 1e-3);
      Assert.assertEquals(marginals.weights[model.t(0, 1)], 0.25 * (T-1), 1e-3);
      Assert.assertEquals(marginals.weights[model.t(1, 0)], 0.25 * (T-1), 1e-3);
      Assert.assertEquals(marginals.weights[model.t(1, 1)], 0.25 * (T-1), 1e-3);
    }

    {
      model.L = 2;
      Parameters marginals = (Parameters) model.getMarginals(model1);
      model.L = 3;
      Assert.assertEquals(marginals.weights[model.o(0, 0)], 0.291, 1e-1 );
      Assert.assertEquals(marginals.weights[model.o(0, 1)], 0.581, 1e-1 );
      Assert.assertEquals(marginals.weights[model.o(1, 0)], 0.564, 1e-1 );
      Assert.assertEquals(marginals.weights[model.o(1, 1)], 0.564, 1e-1 );

      Assert.assertEquals(marginals.weights[model.t(0, 0)], 0.164, 1e-1);
      Assert.assertEquals(marginals.weights[model.t(0, 1)], 0.218, 1e-1);
      Assert.assertEquals(marginals.weights[model.t(1, 0)], 0.327, 1e-1);
      Assert.assertEquals(marginals.weights[model.t(1, 1)], 0.291, 1e-1);
    }
    {
      Example ex_ = new Example(new int[]{0,0}, new int[]{0,0});
      Parameters marginals = (Parameters) model.getMarginals(model1, ex_);
      Assert.assertEquals(marginals.weights[model.o(0, 0)], 0.7, 1e-1 );
      Assert.assertEquals(marginals.weights[model.o(0, 1)], 0.0, 1e-1 );
      Assert.assertEquals(marginals.weights[model.o(1, 0)], 1.3, 1e-1 );
      Assert.assertEquals(marginals.weights[model.o(1, 1)], 0.0, 1e-1 );

      Assert.assertEquals(marginals.weights[model.t(0, 0)], 0.1, 1e-1);
      Assert.assertEquals(marginals.weights[model.t(0, 1)], 0.2, 1e-1);
      Assert.assertEquals(marginals.weights[model.t(1, 0)], 0.3, 1e-1);
      Assert.assertEquals(marginals.weights[model.t(1, 1)], 0.4, 1e-1);
    }
  }

  @Test
  public void testSample() {
    int D = model.D; int K = model.K; int T = model.L;
    {
      // Equal distribution
      Counter<Example> examples = model.drawSamples(model0, new Random(1), 10000);
      for( Example ex : examples ) {
        double fraction = examples.getFraction(ex);
        Assert.assertEquals(1./examples.size(), fraction, 1e-2);
      }
    }
    {
      Counter<Example> examples = model.drawSamples(model1, new Random(1), 1000);
    }
  }

  @Test
  public void testMarginalGradient() {
    double eps = 1e-4;
    // Is the gradent of likelihood this?
    Parameters params = (Parameters) model1.copy();

    Parameters marginal = (Parameters) model.getMarginals(params);
    for(int i = 0; i < marginal.weights.length; i++) {
      params.weights[i] += eps;
      double valuePlus = model.getLogLikelihood(params);
      params.weights[i] -= 2*eps;
      double valueMinus = model.getLogLikelihood(params);
      params.weights[i] += eps;

      double expectedGradient = (valuePlus - valueMinus)/(2*eps);
      double actualGradient = marginal.weights[i];

      Assert.assertTrue( Math.abs(expectedGradient - actualGradient) < 1e-4);
    }
  }

  @Test
  public void testSampleMarginals() {
    Counter<Example> data = model.drawSamples(model1, new Random(1), 1000000);
    Parameters marginal = (Parameters) model.getMarginals(model1, data);
    Parameters marginal_ = model.getSampleMarginals(data);
    for(int i = 0; i < marginal.weights.length; i++) {
      Assert.assertTrue(Math.abs(marginal.weights[i] - marginal_.weights[i]) < 1e-1);
    }
  }

  @Test
  public void testSample2() {
    Counter<Example> data = model.drawSamples(model1, new Random(1), 1000000);
    Parameters marginal = (Parameters) model.getMarginals(model1);
    Parameters marginal_ = model.getSampleMarginals(data);
    for(int i = 0; i < marginal.weights.length; i++) {
      Assert.assertTrue(Math.abs(marginal.weights[i] - marginal_.weights[i]) < 1e-1);
    }
  }

  @Test
  public void testCaching() {
    int D = model.D; int K = model.K; int T = model.L;
    Example ex = new Example(new int[T], new int[T]);

    for(int t = 0; t < T; t++) {
      for(int y_ = 0; y_ < K; y_++) {
        for(int y = 0; y < K; y++) {
          for(int x = 0; x < K; x++) {
            ex.x[t] = x;
            Assert.assertEquals( model1.G(t,y_,y,ex), model1.G_(t,y_,y,ex), 1e-2);
          }
        }
      }
    }
    model1.cache();
    for(int t = 0; t < T; t++) {
      for(int y_ = 0; y_ < K; y_++) {
        for(int y = 0; y < K; y++) {
          for(int x = 0; x < K; x++) {
            ex.x[t] = x;
            Assert.assertEquals( model1.G_(t,y_,y,ex), model1.G(t,y_,y,ex), 1e-2);
          }
        }
      }
    }
    model1.invalidateCache();


  }

}
