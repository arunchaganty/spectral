package learning.models.loglinear;

import junit.framework.Assert;
import learning.linalg.MatrixOps;
import learning.utils.Counter;
import org.junit.Test;

import java.util.Random;

/**
 * Test that models are correctly implemented
 */
public class ModelsTest {

//  @Test
//  public void testMixture() {
//    int L = 1;
//    Models.MixtureModel model = new Models.MixtureModel(2,2,L);
//
//    // At this point, counts should be 1/4 each.
//    {
//      ParamsVec params = model.newParams();
//      ParamsVec counts = model.getMarginals(params);
//      Assert.assertTrue(MatrixOps.allclose(counts.weights, new double[]{0.25, 0.25, 0.25, 0.25}));
//    }
//    {
//      ParamsVec params = model.newParams();
//      params.weights[ model.featureIndexer.getIndex(new UnaryFeature(0, "x="+0)) ] = Math.log(2.0);
//      params.weights[ model.featureIndexer.getIndex(new UnaryFeature(0, "x="+1)) ] = Math.log(3.0);
//      params.weights[ model.featureIndexer.getIndex(new UnaryFeature(1, "x="+0)) ] = Math.log(3.0);
//      params.weights[ model.featureIndexer.getIndex(new UnaryFeature(1, "x="+1)) ] = Math.log(2.0);
//
//      ParamsVec counts = model.getMarginals(params);
//      Assert.assertTrue(MatrixOps.allclose(counts.weights, new double[]{0.2, 0.3, 0.3, 0.2}));
//    }
//  }
//
//  @Test
//  public void testHMM() {
//    int L = 2;
//    Models.HiddenMarkovModel model = new Models.HiddenMarkovModel(2,2,L);
//
//    final ParamsVec params0 = model.newParams();
//    final ParamsVec params1 = model.newParams();
//    {
//      params1.set(new UnaryFeature(0, "x="+0), Math.log(1.) );
//      params1.set(new UnaryFeature(0, "x="+1), Math.log(2.) );
//      params1.set(new UnaryFeature(1, "x="+0), Math.log(1.) );
//      params1.set(new UnaryFeature(1, "x="+1), Math.log(1.) );
//
//      params1.set(new BinaryFeature(0, 0), Math.log(1.) );
//      params1.set(new BinaryFeature(0, 1), Math.log(2.) );
//      params1.set(new BinaryFeature(1, 0), Math.log(3.) );
//      params1.set(new BinaryFeature(1, 1), Math.log(4.) );
//    }
//
//    // At this point, counts should be 1/4 each.
//    {
//      ParamsVec counts = model.getMarginals(params0);
//
//      Assert.assertEquals(counts.get(new UnaryFeature(0, "x="+0) ), 0.5, 1e-3 );
//      Assert.assertEquals(counts.get(new UnaryFeature(0, "x="+1) ), 0.5, 1e-3 );
//      Assert.assertEquals(counts.get(new UnaryFeature(1, "x="+0) ), 0.5, 1e-3 );
//      Assert.assertEquals(counts.get(new UnaryFeature(1, "x="+1) ), 0.5, 1e-3 );
//
//      Assert.assertEquals(counts.get(new BinaryFeature(0, 0) ), 0.25, 1e-3);
//      Assert.assertEquals(counts.get(new BinaryFeature(0, 1) ), 0.25, 1e-3);
//      Assert.assertEquals(counts.get(new BinaryFeature(1, 0) ), 0.25, 1e-3);
//      Assert.assertEquals(counts.get(new BinaryFeature(1, 1) ), 0.25, 1e-3);
//    }
//    {
//      ParamsVec counts = model.getMarginals(params1);
//
//      Assert.assertEquals(counts.get(new UnaryFeature(0, "x="+0) ), 0.2909090909090909, 1e-3 );
//      Assert.assertEquals(counts.get(new UnaryFeature(0, "x="+1) ), 0.5818181818181818, 1e-3 );
//      Assert.assertEquals(counts.get(new UnaryFeature(1, "x="+0) ), 0.5636363636363636, 1e-3 );
//      Assert.assertEquals(counts.get(new UnaryFeature(1, "x="+1) ), 0.5636363636363636, 1e-3 );
//
//      Assert.assertEquals(counts.get(new BinaryFeature(0, 0) ), 0.16363636363636364, 1e-3);
//      Assert.assertEquals(counts.get(new BinaryFeature(0, 1) ), 0.21818181818181817, 1e-3);
//      Assert.assertEquals(counts.get(new BinaryFeature(1, 0) ), 0.3272727272727272, 1e-3);
//      Assert.assertEquals(counts.get(new BinaryFeature(1, 1) ), 0.2909090909090909, 1e-3);
//    }
//
//    {
//      Example ex = new Example(new int[]{0,0}, new int[]{0,0});
//      ParamsVec counts = model.getMarginals(params1, ex);
//
//      Assert.assertEquals(counts.get(new UnaryFeature(0, "x="+0) ), 0.7, 1e-3 );
//      Assert.assertEquals(counts.get(new UnaryFeature(0, "x="+1) ), 0.0, 1e-3 );
//      Assert.assertEquals(counts.get(new UnaryFeature(1, "x="+0) ), 1.3, 1e-3 );
//      Assert.assertEquals(counts.get(new UnaryFeature(1, "x="+1) ), 0.0, 1e-3 );
//
//      Assert.assertEquals(counts.get(new BinaryFeature(0, 0) ), 0.1, 1e-3);
//      Assert.assertEquals(counts.get(new BinaryFeature(0, 1) ), 0.2, 1e-3);
//      Assert.assertEquals(counts.get(new BinaryFeature(1, 0) ), 0.3, 1e-3);
//      Assert.assertEquals(counts.get(new BinaryFeature(1, 1) ), 0.4, 1e-3);
//    }
//
//    {
//      Counter<Example> data = model.drawSamples(params1, new Random(1), 1000000);
//      ParamsVec marginal = model.getMarginals(params1, data);
//      ParamsVec marginal_ = model.getSampleMarginals(data);
//      for(int i = 0; i < marginal.weights.length; i++) {
//        Assert.assertTrue(Math.abs(marginal.weights[i] - marginal_.weights[i]) < 1e-1);
//      }
//    }
//
//    {
//      Counter<Example> data = model.drawSamples(params1, new Random(1), 1000000);
//      ParamsVec marginal = model.getMarginals(params1);
//      ParamsVec marginal_ = model.getSampleMarginals(data);
//      for(int i = 0; i < marginal.weights.length; i++) {
//        Assert.assertTrue(Math.abs(marginal.weights[i] - marginal_.weights[i]) < 1e-1);
//      }
//    }
//
//  }

}
