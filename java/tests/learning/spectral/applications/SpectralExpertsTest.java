package learning.spectral.applications;

import learning.exceptions.RecoveryFailure;
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

    SimpleMatrix Pairs_ = algo.recoverPairs( y, X, reg, true );
    System.out.println( algo.analysis.Pairs );
    System.out.println( Pairs_ );
    algo.analysis.reportPairs(Pairs_);
    Tensor Triples_ = algo.recoverTriples(y, X, reg, true);
    algo.analysis.reportTriples(Triples_);

    Assert.assertTrue( algo.analysis.PairsErr < 1e-2 );
    Assert.assertTrue( algo.analysis.TriplesErr < 1e-1 );
  }

  @Test
  public void testMomentsWithoutBiasEye() {
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.K = 2; options.D = 3; options.betas = "eye"; options.bias = false; options.sigma2 = 0.0;

    MixtureOfExperts model = MixtureOfExperts.generate(options);

    int N = (int) 1e5; double reg = 1e-3;
    testMomentRunner(model, N, reg);
  }
  @Test
  public void testMomentsWithoutBiasRandom() {
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.K = 2; options.D = 3; options.betas = "random"; options.bias = false; options.sigma2 = 0.0;

    MixtureOfExperts model = MixtureOfExperts.generate(options);

    int N = (int) 1e5; double reg = 1e-3;
    testMomentRunner(model, N, reg);
  }
  @Test
  public void testMomentsWithBiasEye() {
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.K = 2; options.D = 1; options.betas = "eye"; options.bias = true; options.sigma2 = 0.0;

    MixtureOfExperts model = MixtureOfExperts.generate(options);

    int N = (int) 1e5; double reg = 1e-2;
    testMomentRunner(model, N, reg);
  }
  @Test
  public void testMomentsWithBiasRandom() {
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.K = 2; options.D = 3; options.betas = "random"; options.bias = true; options.sigma2 = 0.0;

    MixtureOfExperts model = MixtureOfExperts.generate(options);

    int N = (int) 1e5; double reg = 1e-2;
    testMomentRunner(model, N, reg);
  }
  public void arbitraryMomentsTest(MixtureOfExperts.GenerationOptions options) throws NumericalException {

    LogInfo.writeToStdout = true;
    LogInfo.init();

    MixtureOfExperts model = MixtureOfExperts.generate(options);
    SpectralExperts algo = new SpectralExperts();
    algo.enableAnalysis(model);

    // Compute the empirical moments
    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample((int)N);
    SimpleMatrix y = yX.getValue0();
    SimpleMatrix X = yX.getValue1();
    algo.analysis.checkDataSanity(y, X);

//    SimpleMatrix Pairs_ = algo.recoverPairsGD( y, X, reg, doScale );
    SimpleMatrix Pairs_ = algo.recoverPairs( y, X, reg, doScale );
    System.out.println( algo.analysis.Pairs );
    System.out.println( Pairs_ );
    algo.analysis.reportPairs(Pairs_);
    Tensor Triples_ = algo.recoverTriples(y, X, reg, doScale );
    algo.analysis.reportTriples(Triples_);
  }

  public void testRecoveryRunner(MixtureOfExperts model, int N, double reg) throws NumericalException, RecoveryFailure {
    int K = model.getK(); int D = model.getD();
    SpectralExperts algo = new SpectralExperts();
    algo.enableAnalysis(model);

    // Compute the empirical moments
    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample(N);
    SimpleMatrix y = yX.getValue0();
    SimpleMatrix X = yX.getValue1();

    Pair<SimpleMatrix, SimpleMatrix> pair = algo.run(model.getK(), y, X);
    SimpleMatrix weights = pair.getValue0();
    SimpleMatrix betas = pair.getValue1();

    algo.analysis.reportWeights(weights);
    algo.analysis.reportBetas(betas);
    Assert.assertTrue( algo.analysis.betasErr < 1e-1 );
    Assert.assertTrue( algo.analysis.weightsErr < 1e-1 );
  }
  @Test
  public void testRecoveryWithoutBiasEye() throws NumericalException, RecoveryFailure {
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.K = 2; options.D = 3; options.betas = "eye"; options.bias = false; options.sigma2 = 0.0;

    MixtureOfExperts model = MixtureOfExperts.generate(options);

    int N = (int) 1e5; double reg = 1e-3;
    testRecoveryRunner(model, N, reg);
  }
  @Test
  public void testRecoveryWithoutBiasRandom() throws NumericalException, RecoveryFailure {
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.K = 2; options.D = 3; options.betas = "random"; options.bias = false; options.sigma2 = 0.0;

    MixtureOfExperts model = MixtureOfExperts.generate(options);

    int N = (int) 1e5; double reg = 1e-3;
    testRecoveryRunner(model, N, reg);
  }
  @Test
  public void testRecoveryWithBiasEye() throws NumericalException, RecoveryFailure {
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.K = 2; options.D = 1; options.betas = "eye"; options.bias = true; options.sigma2 = 0.0;

    MixtureOfExperts model = MixtureOfExperts.generate(options);

    int N = (int) 1e5; double reg = 1e-2;
    testRecoveryRunner(model, N, reg);
  }
  @Test
  public void testRecoveryWithBiasRandom() throws NumericalException, RecoveryFailure {
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.K = 2; options.D = 3; options.betas = "random"; options.bias = true; options.sigma2 = 0.0;

    MixtureOfExperts model = MixtureOfExperts.generate(options);

    int N = (int) 1e5; double reg = 1e-2;
    testRecoveryRunner(model, N, reg);
  }
  public void arbitraryRecoveryTest(MixtureOfExperts.GenerationOptions options) throws NumericalException, RecoveryFailure {
    LogInfo.writeToStdout = true;
    LogInfo.init();

    MixtureOfExperts model = MixtureOfExperts.generate(options);
    SpectralExperts algo = new SpectralExperts();
    algo.enableAnalysis(model);

    // Compute the empirical moments
    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample((int)N);
    SimpleMatrix y = yX.getValue0();
    SimpleMatrix X = yX.getValue1();

    Pair<SimpleMatrix, SimpleMatrix> pair = algo.run(model.getK(), y, X);
    SimpleMatrix weights = pair.getValue0();
    SimpleMatrix betas = pair.getValue1();

    algo.analysis.reportWeights(weights);
    algo.analysis.reportBetas(betas);
  }

  @Option( gloss = "Number of samples" )
  public double N = 1e4;
  @Option( gloss = "Regularization" )
  public double reg = 1e-3;
  @Option( gloss = "Scale data?" )
  public boolean doScale = false;
  @Option( gloss = "Test the moments?" )
  public boolean testMoments = false;

  public static void main( String[] args ) throws RecoveryFailure, NumericalException {
    SpectralExpertsTest test = new SpectralExpertsTest();
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    OptionsParser parser = new OptionsParser( test, options );

    if( parser.parse( args ) ) {
      if( test.testMoments )
        test.arbitraryMomentsTest(options);
      else
        test.arbitraryRecoveryTest(options);
    }
  }


}
