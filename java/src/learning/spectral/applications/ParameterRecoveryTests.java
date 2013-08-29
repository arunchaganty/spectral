package learning.spectral.applications;

import fig.basic.LogInfo;
import junit.framework.Assert;
import learning.data.ComputableMoments;
import learning.data.HasExactMoments;
import learning.data.HasSampleMoments;
import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import learning.models.HiddenMarkovModel;
import learning.models.MixtureOfGaussians;
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

  public void compareHMMs(HiddenMarkovModel model, HiddenMarkovModel model_, double eps, boolean makeAssertions) {
    SimpleMatrix O = model.getO();
    SimpleMatrix O_ = model_.getO();
    int[] perm = MatrixOps.alignColumns(O_, O);
    O_ = MatrixOps.permuteColumns(O_, perm);

    // Compare that the pi, T and O match
    SimpleMatrix pi = model.getPi();
    SimpleMatrix pi_ = model_.getPi();
    pi_ = MatrixOps.permuteRows(pi_, perm);

    // Because of label permutation freedom, we can have permutation between rows and columns.
    SimpleMatrix T = model.getT();
    SimpleMatrix T_ = model_.getT();
    T_ = MatrixOps.permuteColumns(T_, perm);
    T_ = MatrixOps.permuteRows(T_, perm);

    LogInfo.logs("pi error: " + MatrixOps.diff(pi, pi_));
    LogInfo.logs("O error: " + MatrixOps.diff(O, O_));
    LogInfo.logs("T error: " + MatrixOps.diff(T, T_));

    if(makeAssertions) {
      Assert.assertTrue( MatrixOps.allclose( O, O_, eps) );
      Assert.assertTrue( MatrixOps.allclose( pi, pi_, eps) );
      Assert.assertTrue( MatrixOps.allclose( T, T_, eps) );
    }
  }
  public void compareGMMs(MixtureOfGaussians model, MixtureOfGaussians model_, double eps, boolean makeAssertions) {
    SimpleMatrix pi = model.getWeights();
    SimpleMatrix M1 = model.getMeans()[0];
    SimpleMatrix M2 = model.getMeans()[1];
    SimpleMatrix M3 = model.getMeans()[2];

    SimpleMatrix pi_ = model_.getWeights();
    SimpleMatrix M1_ = model_.getMeans()[0];
    SimpleMatrix M2_ = model_.getMeans()[1];
    SimpleMatrix M3_ = model_.getMeans()[2];

    int[] perm = MatrixOps.alignColumns(M1_, M1);
    pi_ = MatrixOps.permuteColumns(pi_, perm);
    M1_ = MatrixOps.permuteColumns(M1_, perm);
    M2_ = MatrixOps.permuteColumns(M2_, perm);
    M3_ = MatrixOps.permuteColumns(M3_, perm);

    LogInfo.logs("pi error: " + MatrixOps.diff(pi, pi_));
    LogInfo.logs("M1 error: " + MatrixOps.diff(M1, M1_));
    LogInfo.logs("M2 error: " + MatrixOps.diff(M2, M2_));
    LogInfo.logs("M3 error: " + MatrixOps.diff(M3, M3_));

    if(makeAssertions) {
      Assert.assertTrue( MatrixOps.allclose( pi, pi_, eps) );
      Assert.assertTrue( MatrixOps.allclose( M1, M1_, eps) );
      Assert.assertTrue( MatrixOps.allclose( M2, M2_, eps) );
      Assert.assertTrue( MatrixOps.allclose( M3, M3_, eps) );
    }
  }

  @Test
  public void testExactHMMRecovery() {
    // Small
    HiddenMarkovModel model, model_;
    {
      int K = 2; int D = 2;
      model = new HiddenMarkovModel(new HiddenMarkovModel.Params(
          new double[]{0.6,0.4},
          new double[][]{{0.4, 0.6}, {0.6,0.4}},
          new double[][]{{1.0, 0.0}, {0.0,1.0}}));
      model_ = ParameterRecovery.recoverHMM(K, model, 0.0);
      compareHMMs(model, model_, 1e-5, true);
    }
    {
      int K = 2; int D = 2;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      ParameterRecovery.hmmAnalysis = new ParameterRecovery.HMMAnalysis(model);
      model_ = ParameterRecovery.recoverHMM(K, model, 0.0);
      compareHMMs(model, model_, 1e-5, true);
    }
    {
      int K = 2; int D = 3;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      ParameterRecovery.hmmAnalysis = new ParameterRecovery.HMMAnalysis(model);
      model_ = ParameterRecovery.recoverHMM(K, model, 0.0);
      compareHMMs(model, model_, 1e-5, true);
    }
    {
      int K = 3; int D = 3;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      ParameterRecovery.hmmAnalysis = new ParameterRecovery.HMMAnalysis(model);
      model_ = ParameterRecovery.recoverHMM(K, model, 0.0);
      compareHMMs(model, model_, 1e-5, true);
    }
  }

  @Test
  public void testSampleHMMRecovery() {
    // Small
    HiddenMarkovModel model, model_;
    {
      int K = 2; int D = 2; int N = (int) 1e6;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      ParameterRecovery.hmmAnalysis = new ParameterRecovery.HMMAnalysis(model);
      model_ = ParameterRecovery.recoverHMM(K, N, model, 0.0);
      compareHMMs(model, model_, 1e-1, false);
    }
    {
      int K = 2; int D = 3; int N = (int) 1e6;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      ParameterRecovery.hmmAnalysis = new ParameterRecovery.HMMAnalysis(model);
      model_ = ParameterRecovery.recoverHMM(K, N, model, 0.0);
      compareHMMs(model, model_, 1e-1, false);
    }
    {
      int K = 3; int D = 3; int N = (int) 1e7;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      ParameterRecovery.hmmAnalysis = new ParameterRecovery.HMMAnalysis(model);
      model_ = ParameterRecovery.recoverHMM(K, N, model, 0.0);
      compareHMMs(model, model_, 1e-1, false);
    }
  }

  @Test
  public void testComputableHMMRecovery() {
    // Small
    HiddenMarkovModel model, model_;
    {
      int K = 2; int D = 2; int N = (int) 1e6;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      model_ = ParameterRecovery.recoverHMM(K, wrap(model.computeSampleMoments(N)), 0.0);
      compareHMMs(model, model_, 1e-1, false);
    }
    {
      int K = 2; int D = 3; int N = (int) 1e6;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      model_ = ParameterRecovery.recoverHMM(K, wrap(model.computeSampleMoments(N)), 0.0);
      compareHMMs(model, model_, 1e-1, false);
    }
    {
      int K = 3; int D = 3; int N = (int) 1e6;
      model = HiddenMarkovModel.generate(new HiddenMarkovModel.GenerationOptions(K, D));
      model_ = ParameterRecovery.recoverHMM(K, wrap(model.computeSampleMoments(N)), 0.0);
      compareHMMs(model, model_, 1e-1, false);
    }
  }

  private ComputableMoments wrap(final Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments) {
    return new ComputableMoments() {
      @Override
      public MatrixOps.Matrixable computeP13() {
        return MatrixOps.matrixable(moments.getValue0());
      }

      @Override
      public MatrixOps.Matrixable computeP12() {
        return MatrixOps.matrixable(moments.getValue1());
      }

      @Override
      public MatrixOps.Matrixable computeP32() {
        return MatrixOps.matrixable(moments.getValue2());
      }

      @Override
      public MatrixOps.Tensorable computeP123() {
        return MatrixOps.tensorable(moments.getValue3());
      }
    };
  }

  @Test
  public void testExactGMMRecovery() {
    // Small
    MixtureOfGaussians model, model_;
    {
      int K = 2; int D = 2;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "identical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, model, 0.0);
      compareGMMs(model, model_, 1e-5, true);
    }
    {
      int K = 2; int D = 3;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "identical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, model, 0.0);
      compareGMMs(model, model_, 1e-5, true);
    }
    {
      int K = 3; int D = 3;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "identical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, model, 0.0);
      compareGMMs(model, model_, 1e-5, true);
    }
  }

  @Test
  public void testSampleGMMRecovery() {
    // Small
    MixtureOfGaussians model, model_;
    {
      int K = 2; int D = 2;  int N = (int) 1e6;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "identical";options.covs = "spherical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, N, model, 0.0);
      LogInfo.logs("K=%d D=%d N=%d identical", K, D, N);
      compareGMMs(model, model_, 1e-5, false);
    }
    {
      int K = 2; int D = 2;  int N = (int) 1e6;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "random";options.covs = "spherical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, N, model, 0.0);
      LogInfo.logs("K=%d D=%d N=%d random", K, D, N);
      compareGMMs(model, model_, 1e-5, false);
    }
    {
      int K = 2; int D = 3;  int N = (int) 1e6;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "identical";options.covs = "spherical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, N, model, 0.0);
      LogInfo.logs("K=%d D=%d N=%d identical", K, D, N);
      compareGMMs(model, model_, 1e-5, false);
    }
    {
      int K = 2; int D = 3;  int N = (int) 1e6;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "random";options.covs = "spherical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, N, model, 0.0);
      LogInfo.logs("K=%d D=%d N=%d random", K, D, N);
      compareGMMs(model, model_, 1e-5, false);
    }
    {
      int K = 3; int D = 3;  int N = (int) 1e6;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "identical";options.covs = "spherical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, N, model, 0.0);
      LogInfo.logs("K=%d D=%d N=%d identical", K, D, N);
      compareGMMs(model, model_, 1e-5, false);
    }
    {
      int K = 3; int D = 3;  int N = (int) 1e6;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "random";options.covs = "spherical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, N, model, 0.0);
      LogInfo.logs("K=%d D=%d N=%d random", K, D, N);
      compareGMMs(model, model_, 1e-5, false);
    }
    {
      int K = 3; int D = 15;  int N = (int) 1e6;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "identical";options.covs = "spherical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, N, model, 0.0);
      LogInfo.logs("K=%d D=%d N=%d identical", K, D, N);
      compareGMMs(model, model_, 1e-5, false);
    }
    {
      int K = 3; int D = 15;  int N = (int) 1e6;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "random"; options.covs = "spherical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, N, model, 0.0);
      LogInfo.logs("K=%d D=%d N=%d random", K, D, N);
      compareGMMs(model, model_, 1e-5, false);
    }
    {
      int K = 6; int D = 15;  int N = (int) 1e6;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "identical";options.covs = "spherical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, N, model, 0.0);
      LogInfo.logs("K=%d D=%d N=%d identical", K, D, N);
      compareGMMs(model, model_, 1e-5, false);
    }
    {
      int K = 6; int D = 15;  int N = (int) 1e6;
      MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();
      options.K = K; options.D = D; options.means = "random"; options.covs = "spherical";
      model = MixtureOfGaussians.generate(options);
      model_ = ParameterRecovery.recoverGMM(K, N, model, 0.0);
      LogInfo.logs("K=%d D=%d N=%d random", K, D, N);
      compareGMMs(model, model_, 1e-5, false);
    }
  }



}
