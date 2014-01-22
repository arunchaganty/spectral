package learning.models.loglinear;

import fig.basic.Fmt;
import fig.basic.LogInfo;
import learning.common.Counter;
import learning.linalg.MatrixOps;
import learning.models.Params;
import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

/**
 * Test whether we've got this right
 */
public class LatentGridModelTest {
  @Test
  public void testGrid() {
    int K = 2;
    int D = 2;
    int L = 4;
    LatentGridModel model = new LatentGridModel(K, D, L);

    final Params params0 = model.newParams();
    final Params params1 = model.newParams();
    {
      params1.set(model.oString(0, 0), Math.log(1.) );
      params1.set(model.oString(0, 1), Math.log(2.) );
      params1.set(model.oString(1, 0), Math.log(1.) );
      params1.set(model.oString(1, 1), Math.log(1.) );

      params1.set(model.tString(0,0), Math.log(1.) );
      params1.set(model.tString(0,1), Math.log(2.) );
      params1.set(model.tString(1,0), Math.log(3.) );
      params1.set(model.tString(1,1), Math.log(4.) );
    }

    // At this point, counts should be 1/4 each.
    {
      Counter<Example> samples = model.drawSamples(params0, new Random(1), (int)1e5);
      double[][] bins = new double[2][2];
      double[][] Xbins = new double[2][2];
      for(Example ex: samples) {
        Xbins[ex.h[model.hIdx(0, 0)]][ex.x[model.oIdx(0, 0, 0)]] += 1. / samples.size();
        Xbins[ex.h[model.hIdx(0, 0)]][ex.x[model.oIdx(0, 0, 1)]] += 1./ samples.size();
        Xbins[ex.h[model.hIdx(0, 1)]][ex.x[model.oIdx(0, 1, 0)]] += 1./ samples.size();
        Xbins[ex.h[model.hIdx(0, 1)]][ex.x[model.oIdx(0, 1, 1)]] += 1./ samples.size();
        Xbins[ex.h[model.hIdx(1, 0)]][ex.x[model.oIdx(1, 0, 0)]] += 1./ samples.size();
        Xbins[ex.h[model.hIdx(1, 0)]][ex.x[model.oIdx(1, 0, 1)]] += 1./ samples.size();
        Xbins[ex.h[model.hIdx(1, 1)]][ex.x[model.oIdx(1, 1, 0)]] += 1./ samples.size();
        Xbins[ex.h[model.hIdx(1, 1)]][ex.x[model.oIdx(1, 1, 1)]] += 1./ samples.size();

        bins[ex.h[model.hIdx(0, 0)]][ex.h[model.hIdx(0, 1)]] += 1./samples.size();
        bins[ex.h[model.hIdx(0, 0)]][ex.h[model.hIdx(1, 0)]] += 1./samples.size();
        bins[ex.h[model.hIdx(0, 1)]][ex.h[model.hIdx(1, 1)]] += 1./samples.size();
        bins[ex.h[model.hIdx(1, 0)]][ex.h[model.hIdx(1, 1)]] += 1./samples.size();
      }
      // Normalize

      Params counts = model.getMarginals(params0);
      LogInfo.log(Fmt.D(Xbins));
      LogInfo.log(Fmt.D(bins));
      LogInfo.log(counts);

      double emissions = 0.;
      for(int h = 0; h < K; h++) {
        for(int x = 0; x < D; x++) {
          emissions += counts.get(model.oString(h,x));
        }
      }
      Assert.assertEquals(2 * L, emissions, 1e-4);

      double transitions = 0.;
      for(int h_ = 0; h_ < K; h_++) {
        for(int h = 0; h < K; h++) {
          transitions += counts.get(model.tString(h_, h));
        }
      }
      Assert.assertEquals(L, transitions, 1e-4);

      Assert.assertEquals(counts.get(model.oString(0,0)), 0.25 * 8, 1e-3);
      Assert.assertEquals(counts.get(model.oString(0,1)), 0.25 * 8, 1e-3);
      Assert.assertEquals(counts.get(model.oString(1,0)), 0.25 * 8, 1e-3);
      Assert.assertEquals(counts.get(model.oString(1,1)), 0.25 * 8, 1e-3);

      Assert.assertEquals(counts.get(model.tString(0,0)), 0.25 * 4, 1e-3);
      Assert.assertEquals(counts.get(model.tString(0,1)), 0.25 * 4, 1e-3);
      Assert.assertEquals(counts.get(model.tString(1,0)), 0.25 * 4, 1e-3);
      Assert.assertEquals(counts.get(model.tString(1,1)), 0.25 * 4, 1e-3);
    }
    {
      double logZ = model.getLogLikelihood(params1, L);
//      Assert.assertEquals(Math.log(266200.), logZ, 1e-2);
      Params counts = model.getMarginals(params1);
      LogInfo.log(counts);


      double emissions = 0.;
      for(int h = 0; h < K; h++) {
        for(int x = 0; x < D; x++) {
          emissions += counts.get(model.oString(h,x));
        }
      }
      Assert.assertEquals(2 * L, emissions, 1e-4);

      double transitions = 0.;
      for(int h_ = 0; h_ < K; h_++) {
        for(int h = 0; h < K; h++) {
          transitions += counts.get(model.tString(h_,h));
        }
      }
      Assert.assertEquals(L, transitions, 1e-4);


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
      LogInfo.log(counts);

      double emissions = 0.;
      for(int h = 0; h < K; h++) {
        for(int x = 0; x < D; x++) {
          emissions += counts.get(model.oString(h,x));
        }
      }
      Assert.assertEquals(2 * L, emissions, 1e-4);

      double transitions = 0.;
      for(int h_ = 0; h_ < K; h_++) {
        for(int h = 0; h < K; h++) {
          transitions += counts.get(model.tString(h_,h));
        }
      }
      Assert.assertEquals(L, transitions, 1e-4);


//      Assert.assertEquals(counts.get(model.oString(0,0)), 0.7 * 8, 1e-3);
//      Assert.assertEquals(counts.get(model.oString(0,1)), 0.0 * 8, 1e-3);
//      Assert.assertEquals(counts.get(model.oString(1,0)), 1.3 * 8, 1e-3);
//      Assert.assertEquals(counts.get(model.oString(1,1)), 0.0 * 8, 1e-3);
//
//      Assert.assertEquals(counts.get("h1=0,h2=0"), 0.1, 1e-3);
//      Assert.assertEquals(counts.get("h1=0,h2=1"), 0.2, 1e-3);
//      Assert.assertEquals(counts.get("h1=1,h2=0"), 0.3, 1e-3);
//      Assert.assertEquals(counts.get("h1=1,h2=1"), 0.4, 1e-3);
    }

    {
      Counter<Example> data = model.drawSamples(params1, new Random(1), 1000000);
      Params marginal = model.getMarginals(params1);
      Params marginal_ = model.getSampleMarginals(data);
      for(int i = 0; i < marginal.size(); i++) {
        Assert.assertTrue(Math.abs(marginal.toArray()[i] - marginal_.toArray()[i]) < 1e-1);
      }
    }

    {
      Counter<Example> data = model.drawSamples(params1, new Random(1), 1000000);
      Params marginal = model.getMarginals(params1, data);
      Params marginal_ = model.getSampleMarginals(data);
      for(int i = 0; i < marginal.size(); i++) {
        Assert.assertTrue(Math.abs(marginal.toArray()[i] - marginal_.toArray()[i]) < 1e-1);
      }
    }

  }

}
