package learning.models.loglinear;

import Jama.Matrix;
import fig.basic.LogInfo;
import learning.common.Counter;
import learning.linalg.MatrixOps;
import learning.models.Params;
import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

/**
 * Test that models are correctly implemented
 */
public class ModelsTest {

  @Test
  public void testMixture() {
    int L = 1;
    Models.MixtureModel model = new Models.MixtureModel(2,2,L);

    // At this point, counts should be 1/4 each.
    {
      Params params = model.newParams();
      Params counts = model.getMarginals(params);
      Assert.assertTrue(MatrixOps.allclose(counts.toArray(), new double[]{0.25, 0.25, 0.25, 0.25}));
    }
    {
      Params params = model.newParams();
      params.set("h=0:x=0", Math.log(2.0));
      params.set("h=0:x=1", Math.log(3.0));
      params.set("h=1:x=0", Math.log(3.0));
      params.set("h=1:x=1", Math.log(2.0));
      Params counts = model.getMarginals(params);
      Assert.assertTrue(MatrixOps.allclose(counts.toArray(), new double[]{0.2, 0.3, 0.3, 0.2}));
    }
  }

  @Test
  public void testHMM() {
    int L = 2;
    Models.HiddenMarkovModel model = new Models.HiddenMarkovModel(2,2,L);

    final Params params0 = model.newParams();
    final Params params1 = model.newParams();
    {
      params1.set("h=0:x="+0, Math.log(1.) );
      params1.set("h=0:x="+1, Math.log(2.) );
      params1.set("h=1:x="+0, Math.log(1.) );
      params1.set("h=1:x="+1, Math.log(1.) );

      params1.set("h1=0,h2=0", Math.log(1.) );
      params1.set("h1=0,h2=1", Math.log(2.) );
      params1.set("h1=1,h2=0", Math.log(3.) );
      params1.set("h1=1,h2=1", Math.log(4.) );
    }

    // At this point, counts should be 1/4 each.
    {
      Params counts = model.getMarginals(params0);

      Assert.assertEquals(counts.get("h=0:x="+0), 0.5, 1e-3 );
      Assert.assertEquals(counts.get("h=0:x="+1), 0.5, 1e-3 );
      Assert.assertEquals(counts.get("h=1:x="+0), 0.5, 1e-3 );
      Assert.assertEquals(counts.get("h=1:x="+1), 0.5, 1e-3 );

      Assert.assertEquals(counts.get("h1=0,h2=0"), 0.25, 1e-3);
      Assert.assertEquals(counts.get("h1=0,h2=1"), 0.25, 1e-3);
      Assert.assertEquals(counts.get("h1=1,h2=0"), 0.25, 1e-3);
      Assert.assertEquals(counts.get("h1=1,h2=1"), 0.25, 1e-3);
    }
    {
      Params counts = model.getMarginals(params1);

      Assert.assertEquals(counts.get("h=0:x="+0), 0.2909090909090909, 1e-3 );
      Assert.assertEquals(counts.get("h=0:x="+1), 0.5818181818181818, 1e-3 );
      Assert.assertEquals(counts.get("h=1:x="+0), 0.5636363636363636, 1e-3 );
      Assert.assertEquals(counts.get("h=1:x="+1), 0.5636363636363636, 1e-3 );

      Assert.assertEquals(counts.get("h1=0,h2=0"), 0.16363636363636364, 1e-3);
      Assert.assertEquals(counts.get("h1=0,h2=1"), 0.21818181818181817, 1e-3);
      Assert.assertEquals(counts.get("h1=1,h2=0"), 0.3272727272727272, 1e-3);
      Assert.assertEquals(counts.get("h1=1,h2=1"), 0.2909090909090909, 1e-3);
    }

    {
      Example ex = new Example(new int[]{0,0}, new int[]{0,0});
      Params counts = model.getMarginals(params1, ex);

      Assert.assertEquals(counts.get("h=0:x="+0), 0.7, 1e-3 );
      Assert.assertEquals(counts.get("h=0:x="+1), 0.0, 1e-3 );
      Assert.assertEquals(counts.get("h=1:x="+0), 1.3, 1e-3 );
      Assert.assertEquals(counts.get("h=1:x="+1), 0.0, 1e-3 );

      Assert.assertEquals(counts.get("h1=0,h2=0"), 0.1, 1e-3);
      Assert.assertEquals(counts.get("h1=0,h2=1"), 0.2, 1e-3);
      Assert.assertEquals(counts.get("h1=1,h2=0"), 0.3, 1e-3);
      Assert.assertEquals(counts.get("h1=1,h2=1"), 0.4, 1e-3);
    }

    {
      Counter<Example> data = model.drawSamples(params1, new Random(1), 1000000);
      Params marginal = model.getMarginals(params1, data);
      Params marginal_ = model.getSampleMarginals(data);
      for(int i = 0; i < marginal.size(); i++) {
        Assert.assertTrue(Math.abs(marginal.toArray()[i] - marginal_.toArray()[i]) < 1e-1);
      }
    }

    {
      Counter<Example> data = model.drawSamples(params1, new Random(1), 1000000);
      Params marginal = model.getMarginals(params1);
      Params marginal_ = model.getSampleMarginals(data);
      for(int i = 0; i < marginal.size(); i++) {
        Assert.assertTrue(Math.abs(marginal.toArray()[i] - marginal_.toArray()[i]) < 1e-1);
      }
    }

  }

  @Test
  public void testGrid() {
    int L = 4;
    Models.GridModel model = new Models.GridModel(2,2,L);

    final Params params0 = model.newParams();
    final Params params1 = model.newParams();
    {
      params1.set("h=0:x="+0, Math.log(1.) );
      params1.set("h=0:x="+1, Math.log(2.) );
      params1.set("h=1:x="+0, Math.log(1.) );
      params1.set("h=1:x="+1, Math.log(1.) );

      params1.set("h1=0,h2=0", Math.log(1.) );
      params1.set("h1=0,h2=1", Math.log(2.) );
      params1.set("h1=1,h2=0", Math.log(3.) );
      params1.set("h1=1,h2=1", Math.log(4.) );
    }

    // At this point, counts should be 1/4 each.
    {
      Counter<Example> samples = model.drawSamples(params0, new Random(1), (int)1e5);
      double[][] bins = new double[2][2];
      double[][] Xbins = new double[2][2];
      for(Example ex: samples) {
        Xbins[ex.h[model.hiddenNodeIndex(0,0)]][ex.x[model.observedNodeIndex(0,0,0)]] += 1.;
        Xbins[ex.h[model.hiddenNodeIndex(0,0)]][ex.x[model.observedNodeIndex(0,0,1)]] += 1.;
        Xbins[ex.h[model.hiddenNodeIndex(0,1)]][ex.x[model.observedNodeIndex(0,1,0)]] += 1.;
        Xbins[ex.h[model.hiddenNodeIndex(0,1)]][ex.x[model.observedNodeIndex(0,1,1)]] += 1.;
        Xbins[ex.h[model.hiddenNodeIndex(1,0)]][ex.x[model.observedNodeIndex(1,0,0)]] += 1.;
        Xbins[ex.h[model.hiddenNodeIndex(1,0)]][ex.x[model.observedNodeIndex(1,0,1)]] += 1.;
        Xbins[ex.h[model.hiddenNodeIndex(1,1)]][ex.x[model.observedNodeIndex(1,1,0)]] += 1.;
        Xbins[ex.h[model.hiddenNodeIndex(1,1)]][ex.x[model.observedNodeIndex(1,1,1)]] += 1.;

        bins[ex.h[model.hiddenNodeIndex(0,0)]][ex.h[model.hiddenNodeIndex(0,1)]] += 1.;
        bins[ex.h[model.hiddenNodeIndex(0,0)]][ex.h[model.hiddenNodeIndex(1,0)]] += 1.;
        bins[ex.h[model.hiddenNodeIndex(0,1)]][ex.h[model.hiddenNodeIndex(1,1)]] += 1.;
        bins[ex.h[model.hiddenNodeIndex(1,0)]][ex.h[model.hiddenNodeIndex(1,1)]] += 1.;
      }
      // Normalize
      double sum;
      sum = MatrixOps.sum(Xbins);
      MatrixOps.scale(Xbins, 1./sum);
      sum = MatrixOps.sum(bins);
      MatrixOps.scale(bins, 1./sum);

      Params counts = model.getMarginals(params0);
      LogInfo.log(counts);


//      Assert.assertEquals(counts.get("h=0:x="+0), 0.25 * 8, 1e-3 );
//      Assert.assertEquals(counts.get("h=0:x="+1), 0.25 * 8, 1e-3 );
//      Assert.assertEquals(counts.get("h=1:x="+0), 0.25 * 8, 1e-3 );
//      Assert.assertEquals(counts.get("h=1:x="+1), 0.25 * 8, 1e-3 );
//
//      Assert.assertEquals(counts.get("h1=0,h2=0"), 0.25 * 4, 1e-3);
//      Assert.assertEquals(counts.get("h1=0,h2=1"), 0.25 * 4, 1e-3);
//      Assert.assertEquals(counts.get("h1=1,h2=0"), 0.25 * 4, 1e-3);
//      Assert.assertEquals(counts.get("h1=1,h2=1"), 0.25 * 4, 1e-3);
    }
    {
      Params counts = model.getMarginals(params1);
      LogInfo.log(counts);

//      Assert.assertEquals(counts.get("h=0:x="+0), 0.2909090909090909, 1e-3 );
//      Assert.assertEquals(counts.get("h=0:x="+1), 0.5818181818181818, 1e-3 );
//      Assert.assertEquals(counts.get("h=1:x="+0), 0.5636363636363636, 1e-3 );
//      Assert.assertEquals(counts.get("h=1:x="+1), 0.5636363636363636, 1e-3 );
//
//      Assert.assertEquals(counts.get("h1=0,h2=0"), 0.16363636363636364, 1e-3);
//      Assert.assertEquals(counts.get("h1=0,h2=1"), 0.21818181818181817, 1e-3);
//      Assert.assertEquals(counts.get("h1=1,h2=0"), 0.3272727272727272, 1e-3);
//      Assert.assertEquals(counts.get("h1=1,h2=1"), 0.2909090909090909, 1e-3);
    }

    {
      Example ex = new Example(new int[]{0,0,0,0,0,0,0,0}, new int[]{0,0,0,0,0,0,0,0});
      Params counts = model.getMarginals(params1, ex);

//      Assert.assertEquals(counts.get("h=0:x="+0), 0.7, 1e-3 );
//      Assert.assertEquals(counts.get("h=0:x="+1), 0.0, 1e-3 );
//      Assert.assertEquals(counts.get("h=1:x="+0), 1.3, 1e-3 );
//      Assert.assertEquals(counts.get("h=1:x="+1), 0.0, 1e-3 );
//
//      Assert.assertEquals(counts.get("h1=0,h2=0"), 0.1, 1e-3);
//      Assert.assertEquals(counts.get("h1=0,h2=1"), 0.2, 1e-3);
//      Assert.assertEquals(counts.get("h1=1,h2=0"), 0.3, 1e-3);
//      Assert.assertEquals(counts.get("h1=1,h2=1"), 0.4, 1e-3);
    }

    {
      Counter<Example> data = model.drawSamples(params1, new Random(1), 1000000);
      Params marginal = model.getMarginals(params1, data);
      Params marginal_ = model.getSampleMarginals(data);
      for(int i = 0; i < marginal.size(); i++) {
        Assert.assertTrue(Math.abs(marginal.toArray()[i] - marginal_.toArray()[i]) < 1e-1);
      }
    }

    {
      Counter<Example> data = model.drawSamples(params1, new Random(1), 1000000);
      Params marginal = model.getMarginals(params1);
      Params marginal_ = model.getSampleMarginals(data);
      for(int i = 0; i < marginal.size(); i++) {
        Assert.assertTrue(Math.abs(marginal.toArray()[i] - marginal_.toArray()[i]) < 1e-1);
      }
    }

  }


}
