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

public class DirectedGridModelTest {

  DirectedGridModel model1;
  DirectedGridModel model2;
  Random testRandom = new Random(42);

  @Before
  public void setup() {
    model1 = new DirectedGridModel(2,2,4);
    model2 = new DirectedGridModel(3,3,4);
  }


  @Test
  public void testSampleMarginals() {
    DirectedGridModel.Parameters params = model1.newParams();
    params.initRandom(testRandom, 1.0);

    Counter<Example> data = model1.drawSamples(params, new Random(1), 1000000);
    DirectedGridModel.Parameters marginal = (DirectedGridModel.Parameters) model1.getSampleMarginals(data);
    for(int i = 0; i < marginal.weights.length; i++) {
      Assert.assertTrue(Math.abs(marginal.weights[i] - params.weights[i]) < 1e-1);
    }
  }
}
