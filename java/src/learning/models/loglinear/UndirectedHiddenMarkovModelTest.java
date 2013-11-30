package learning.models.loglinear;

import fig.basic.Pair;
import learning.utils.Counter;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static learning.models.loglinear.UndirectedHiddenMarkovModel.o;
import static learning.models.loglinear.UndirectedHiddenMarkovModel.t;

/**
 *
 */
public class UndirectedHiddenMarkovModelTest {
  UndirectedHiddenMarkovModel model;
  ParamsVec model0;
  ParamsVec model1;

  @Before
  public void initialize() {
    int K = 2, D = 2, L = 3;
    model = new UndirectedHiddenMarkovModel(K, D, L);
    model0 = model.newParamsVec();

    model1 = model.newParamsVec();
    model1.set(new UnaryFeature(0, "x=" + 0), Math.log(1.));
    model1.set(new UnaryFeature(0, "x=" + 1), Math.log(2.));
    model1.set(new UnaryFeature(1, "x=" + 0), Math.log(1.));
    model1.set(new UnaryFeature(1, "x=" + 1), Math.log(1.));

    model1.set(new BinaryFeature(0, 0), Math.log(1.));
    model1.set(new BinaryFeature(0, 1), Math.log(2.));
    model1.set(new BinaryFeature(1, 0), Math.log(3.));
    model1.set(new BinaryFeature(1, 1), Math.log(4.));
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
          Assert.assertEquals(1. / K, forwards[t][y], 1e-4);
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
      Assert.assertArrayEquals(forwards[0], new double[]{0.6, 0.4}, 1e-2);
      Assert.assertArrayEquals(forwards[1], new double[]{0.49, 0.51}, 1e-2);
      Assert.assertArrayEquals(forwards[2], new double[]{0.501, 0.499}, 1e-2);
    }
    //

    {
      Example ex = new Example(new int[]{0,0,0}, new int[]{0,0,0});
      T = ex.x.length;
      Pair<Double, double[][]> Zforwards = model.forward(model1, ex);
      double Z = Zforwards.getFirst(); double[][] forwards = Zforwards.getSecond();
      Assert.assertEquals(Math.log(54), Z, 1e-4);
      Assert.assertArrayEquals(forwards[0], new double[]{0.5, 0.5}, 1e-2);
      Assert.assertArrayEquals(forwards[1], new double[]{0.4, 0.6}, 1e-2);
      Assert.assertArrayEquals(forwards[2], new double[] { 0.407, 0.593 }, 1e-2);
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
          Assert.assertEquals(1. / K, backwards[t][y], 1e-4);
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
          Assert.assertEquals(1. / K, backwards[t][y], 1e-4);
        }
      }
    }

    // The second model, things have non-unit weights.
    {
      double[][] backwards = model.backward(model1, null);
      Assert.assertArrayEquals(new double[]{0.5, 0.5}, backwards[2], 1e-2);
      Assert.assertArrayEquals(new double[]{0.292, 0.708}, backwards[1], 1e-2);
      Assert.assertArrayEquals(new double[]{0.309, 0.691 }, backwards[0], 1e-2);
    }
    //

    {
      Example ex = new Example(new int[]{0,0,0}, new int[]{0,0,0});
      T = ex.x.length;
      double[][] backwards = model.backward(model1, ex);
      Assert.assertArrayEquals(new double[]{0.5, 0.5}, backwards[2], 1e-2);
      Assert.assertArrayEquals(new double[]{0.3, 0.7}, backwards[1], 1e-2);
      Assert.assertArrayEquals(new double[]{0.314, 0.686 }, backwards[0], 1e-2);
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
      ParamsVec marginals = model.getMarginals(model0);
      int T = model.L;
      Assert.assertEquals(marginals.get(o(0, 0)), 0.25 * T, 1e-3 );
      Assert.assertEquals(marginals.get(o(0, 1)), 0.25 * T, 1e-3 );
      Assert.assertEquals(marginals.get(o(1, 0)), 0.25 * T, 1e-3 );
      Assert.assertEquals(marginals.get(o(1, 1)), 0.25 * T, 1e-3 );

      Assert.assertEquals(marginals.get(t(0, 0)), 0.25 * (T-1), 1e-3);
      Assert.assertEquals(marginals.get(t(0, 1)), 0.25 * (T-1), 1e-3);
      Assert.assertEquals(marginals.get(t(1, 0)), 0.25 * (T-1), 1e-3);
      Assert.assertEquals(marginals.get(t(1, 1)), 0.25 * (T-1), 1e-3);
    }
    {
      int T = model.L;
      ParamsVec marginals = model.getMarginals(model0, ex);
      Assert.assertEquals(marginals.get(o(0, 0)), 0.5 * T, 1e-3 );
      Assert.assertEquals(marginals.get(o(0, 1)), 0.0 * T, 1e-3 );
      Assert.assertEquals(marginals.get(o(1, 0)), 0.5 * T, 1e-3 );
      Assert.assertEquals(marginals.get(o(1, 1)), 0.0 * T, 1e-3 );

      Assert.assertEquals(marginals.get(t(0, 0)), 0.25 * (T-1), 1e-3);
      Assert.assertEquals(marginals.get(t(0, 1)), 0.25 * (T-1), 1e-3);
      Assert.assertEquals(marginals.get(t(1, 0)), 0.25 * (T-1), 1e-3);
      Assert.assertEquals(marginals.get(t(1, 1)), 0.25 * (T-1), 1e-3);
    }

    {
      ParamsVec marginals = model.getMarginals(model1);
      Assert.assertEquals(marginals.get(o(0, 0)), 0.45, 1e-1 );
      Assert.assertEquals(marginals.get(o(0, 1)), 0.90, 1e-1 );
      Assert.assertEquals(marginals.get(o(1, 0)), 0.82, 1e-1 );
      Assert.assertEquals(marginals.get(o(1, 1)), 0.82, 1e-1 );

      Assert.assertEquals(marginals.get(t(0, 0)), 0.21, 1e-1);
      Assert.assertEquals(marginals.get(t(0, 1)), 0.469, 1e-1);
      Assert.assertEquals(marginals.get(t(1, 0)), 0.568, 1e-1);
      Assert.assertEquals(marginals.get(t(1, 1)), 0.745, 1e-1);
    }
    {
      ParamsVec marginals = model.getMarginals(model1, ex);
      Assert.assertEquals(marginals.get(o(0, 0)), 0.94, 1e-1 );
      Assert.assertEquals(marginals.get(o(0, 1)), 0.0, 1e-1 );
      Assert.assertEquals(marginals.get(o(1, 0)), 2.05, 1e-1 );
      Assert.assertEquals(marginals.get(o(1, 1)), 0.0, 1e-1 );

      Assert.assertEquals(marginals.get(t(0, 0)), 0.13, 1e-1);
      Assert.assertEquals(marginals.get(t(0, 1)), 0.407, 1e-1);
      Assert.assertEquals(marginals.get(t(1, 0)), 0.5, 1e-1);
      Assert.assertEquals(marginals.get(t(1, 1)), 0.963, 1e-1);
    }
  }

  @Test
  public void testSample() {
    int D = model.D; int K = model.K; int T = model.L;
    {
      // Equal distribution
      Counter<Example> examples = model.drawSamples(model0, new Random(1), 10000);
      for( Example ex : examples ) {
        double fraction = examples.getCount(ex) / examples.sum();
        Assert.assertEquals(1./examples.size(), fraction, 1e-2);
      }
    }
    {
      Counter<Example> examples = model.drawSamples(model1, new Random(1), 1000);
    }
  }

}
