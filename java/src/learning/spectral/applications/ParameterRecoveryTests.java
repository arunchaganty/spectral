package learning.spectral.applications;

import fig.basic.LogInfo;
import junit.framework.Assert;
import learning.data.HasExactMoments;
import learning.data.HasSampleMoments;
import learning.linalg.MatrixOps;
import learning.models.HiddenMarkovModel;
import learning.spectral.TensorMethod;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;
import org.junit.Test;

/**
 * Tests for parameter recovery
 */
public class ParameterRecoveryTests {
  TensorMethod tensorMethod = new TensorMethod();

  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> recoverExact(int K, HasExactMoments obj) {
    return tensorMethod.recoverParameters(K, obj.computeExactMoments());
  }
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> recoverSample(int K, HasSampleMoments obj, int N) {
    return tensorMethod.recoverParameters(K, obj.computeSampleMoments(N));
  }

  public void compareHMMs(HiddenMarkovModel model, HiddenMarkovModel model_, double eps) {
    // Compare that the pi, T and O match
    SimpleMatrix pi = model.getPi();
    SimpleMatrix pi_ = model_.getPi();
    pi_ = MatrixOps.alignMatrix(pi_, pi, true);

    SimpleMatrix T = model.getT();
    SimpleMatrix T_ = model_.getT();
    T_ = MatrixOps.alignMatrix(T_, T, true);

    SimpleMatrix O = model.getO();
    SimpleMatrix O_ = model_.getO();
    O_ = MatrixOps.alignMatrix(O_, O, true);

    LogInfo.logs("pi error: " + MatrixOps.diff(pi, pi_));
    LogInfo.logs("O error: " + MatrixOps.diff(T, T_));
    LogInfo.logs("T error: " + MatrixOps.diff(O, O_));

    Assert.assertTrue( MatrixOps.allclose( O, O_, eps) );
    Assert.assertTrue( MatrixOps.allclose( pi, pi_, eps) );
    Assert.assertTrue( MatrixOps.allclose( T, T_, eps) );
  }

  @Test
  public void testExactHMMRecovery() {
    // Small
    HiddenMarkovModel model, model_;
    {
      int K = 2; int D = 2;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      model_ = ParameterRecovery.recoverHMM(K, model, 0.0);
      compareHMMs(model, model_, 1e-5);
    }
    {
      int K = 2; int D = 3;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      model_ = ParameterRecovery.recoverHMM(K, model, 0.0);
      compareHMMs(model, model_, 1e-5);
    }
    {
      int K = 3; int D = 3;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      model_ = ParameterRecovery.recoverHMM(K, model, 0.0);
      compareHMMs(model, model_, 1e-5);
    }
  }

  @Test
  public void testSampleHMMRecovery() {
    // Small
    HiddenMarkovModel model, model_;
    {
      int K = 2; int D = 2; int N = (int) 1e5;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      model_ = ParameterRecovery.recoverHMM(K, N, model, 0.0);
      compareHMMs(model, model_, 1e-1);
    }
    {
      int K = 2; int D = 3; int N = (int) 1e5;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      model_ = ParameterRecovery.recoverHMM(K, N, model, 0.0);
      compareHMMs(model, model_, 1e-1);
    }
    {
      int K = 3; int D = 3; int N = (int) 1e5;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      model_ = ParameterRecovery.recoverHMM(K, N, model, 0.0);
      compareHMMs(model, model_, 1e-1);
    }
  }


}
