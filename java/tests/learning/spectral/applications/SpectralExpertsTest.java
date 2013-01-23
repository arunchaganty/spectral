package learning.spectral.applications;

import learning.linalg.*;
import learning.exceptions.NumericalException;

import learning.spectral.applications.SpectralExperts;

import learning.models.MixtureOfExperts;

import org.javatuples.*;
import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionsParser;

/**
 * Tests for spectral experts
 */
public class SpectralExpertsTest {

  public void testMomentRunner(MixtureOfExperts model, int N, double reg) {
    int K = model.getK(); int D = model.getD();
    SpectralExperts algo = new SpectralExperts();
    algo.enableAnalysis(model);

    // Compute the empirical moments
    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample(N);
    SimpleMatrix y = yX.getValue0();
    SimpleMatrix X = yX.getValue1();

    SimpleMatrix Pairs_ = algo.recoverPairs( y, X, reg );
    algo.analysis.reportPairs(Pairs_);
    Tensor Triples_ = algo.recoverTriples(y, X, reg);
    algo.analysis.reportTriples(Triples_);

    Assert.assertTrue( algo.analysis.PairsErr < 1e-2 );
    Assert.assertTrue( algo.analysis.TriplesErr < 1e-2 );
  }

  @Test
  public void testMomentsWithoutBias() {
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.K = 2; options.D = 3; options.betas = "random"; options.bias = false; options.sigma2 = 0.0;

    MixtureOfExperts model = MixtureOfExperts.generate(options);

    int N = (int) 1e5; double reg = 1e-3;
    testMomentRunner(model, N, reg);
  }

  //@Test
  public void testMomentsWithBias() {
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.K = 2; options.D = 1; options.betas = "random"; options.bias = true; options.sigma2 = 0.0;

    MixtureOfExperts model = MixtureOfExperts.generate(options);

    int N = (int) 1e5; double reg = 1e-2;
    testMomentRunner(model, N, reg);
  }

}
