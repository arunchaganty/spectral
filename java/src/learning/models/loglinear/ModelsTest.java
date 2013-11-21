package learning.models.loglinear;

import fig.basic.Hypergraph;
import fig.basic.LogInfo;
import junit.framework.Assert;
import learning.linalg.MatrixOps;
import org.junit.Test;

/**
 * Test that models are correctly implemented
 */
public class ModelsTest {

  @Test
  public void testMixture() {
    int L = 1;
    Models.MixtureModel model = new Models.MixtureModel(2,2,L);
    model.createHypergraph(null, null, 0.0);

    // At this point, counts should be 1/4 each.
    {
      ParamsVec params = model.newParamsVec();
      ParamsVec counts = model.newParamsVec();
      Hypergraph<Example> Hp = model.createHypergraph(params.weights, counts.weights, 1.0);
      Hp.computePosteriors(false);
      Hp.fetchPosteriors(false);
      Assert.assertTrue(MatrixOps.allclose(counts.weights, new double[]{0.25, 0.25, 0.25, 0.25}));
    }
    {
      ParamsVec params = model.newParamsVec();
      params.weights[ model.featureIndexer.getIndex(new UnaryFeature(0, "x="+0)) ] = Math.log(2.0);
      params.weights[ model.featureIndexer.getIndex(new UnaryFeature(0, "x="+1)) ] = Math.log(3.0);
      params.weights[ model.featureIndexer.getIndex(new UnaryFeature(1, "x="+0)) ] = Math.log(3.0);
      params.weights[ model.featureIndexer.getIndex(new UnaryFeature(1, "x="+1)) ] = Math.log(2.0);

      ParamsVec counts = model.newParamsVec();
      Hypergraph<Example> Hp = model.createHypergraph(params.weights, counts.weights, 1.0);
      Hp.computePosteriors(false);
      Hp.fetchPosteriors(false);

      Assert.assertTrue(MatrixOps.allclose(counts.weights, new double[]{0.2, 0.3, 0.3, 0.2}));
    }
  }

  @Test
  public void testHMM() {
    int L = 2;
    Models.HiddenMarkovModel model = new Models.HiddenMarkovModel(2,2,L);
    model.createHypergraph(null, null, 0.0);

    // At this point, counts should be 1/4 each.
    {
      ParamsVec params = model.newParamsVec();
      ParamsVec counts = model.newParamsVec();
      Hypergraph<Example> Hp = model.createHypergraph(params.weights, counts.weights, 1.0);
      Hp.computePosteriors(false);
      Hp.fetchPosteriors(false);

      Assert.assertEquals(counts.get(new UnaryFeature(0, "x="+0) ), 0.5, 1e-3 );
      Assert.assertEquals(counts.get(new UnaryFeature(0, "x="+1) ), 0.5, 1e-3 );
      Assert.assertEquals(counts.get(new UnaryFeature(1, "x="+0) ), 0.5, 1e-3 );
      Assert.assertEquals(counts.get(new UnaryFeature(1, "x="+1) ), 0.5, 1e-3 );

      Assert.assertEquals(counts.get(new BinaryFeature(0, 0) ), 0.25, 1e-3);
      Assert.assertEquals(counts.get(new BinaryFeature(0, 1) ), 0.25, 1e-3);
      Assert.assertEquals(counts.get(new BinaryFeature(1, 0) ), 0.25, 1e-3);
      Assert.assertEquals(counts.get(new BinaryFeature(1, 1) ), 0.25, 1e-3);
    }
    {
      ParamsVec params = model.newParamsVec();
      params.set(new UnaryFeature(0, "x="+0), Math.log(1.) );
      params.set(new UnaryFeature(0, "x="+1), Math.log(2.) );
      params.set(new UnaryFeature(1, "x="+0), Math.log(1.) );
      params.set(new UnaryFeature(1, "x="+1), Math.log(1.) );

      params.set(new BinaryFeature(0, 0), Math.log(1.) );
      params.set(new BinaryFeature(0, 1), Math.log(2.) );
      params.set(new BinaryFeature(1, 0), Math.log(3.) );
      params.set(new BinaryFeature(1, 1), Math.log(4.) );

      ParamsVec counts = model.newParamsVec();
      Hypergraph<Example> Hp = model.createHypergraph(params.weights, counts.weights, 1.0);
      Hp.computePosteriors(false);
      Hp.fetchPosteriors(false);

      Assert.assertEquals(counts.get(new UnaryFeature(0, "x="+0) ), 0.2909090909090909, 1e-3 );
      Assert.assertEquals(counts.get(new UnaryFeature(0, "x="+1) ), 0.5818181818181818, 1e-3 );
      Assert.assertEquals(counts.get(new UnaryFeature(1, "x="+0) ), 0.5636363636363636, 1e-3 );
      Assert.assertEquals(counts.get(new UnaryFeature(1, "x="+1) ), 0.5636363636363636, 1e-3 );

      Assert.assertEquals(counts.get(new BinaryFeature(0, 0) ), 0.16363636363636364, 1e-3);
      Assert.assertEquals(counts.get(new BinaryFeature(0, 1) ), 0.21818181818181817, 1e-3);
      Assert.assertEquals(counts.get(new BinaryFeature(1, 0) ), 0.3272727272727272, 1e-3);
      Assert.assertEquals(counts.get(new BinaryFeature(1, 1) ), 0.2909090909090909, 1e-3);
    }

  }

}
