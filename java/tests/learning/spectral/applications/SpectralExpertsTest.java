package learning.spectral.applications;

import learning.exceptions.RecoveryFailure;
import learning.linalg.*;
import learning.exceptions.NumericalException;

import learning.spectral.applications.SpectralExperts;

import learning.models.MixtureOfExperts;

import org.javatuples.*;
import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;
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

  @Before
  public void setup(){
    LogInfo.writeToStdout = false;
    LogInfo.init();
  }

  public void testRegressionRunner(MixtureOfExperts model, int N, double reg) {
    int K = model.getK(); int D = model.getD();
    SpectralExperts algo = new SpectralExperts();
    algo.K = K;
    algo.enableAnalysis(model);

    // Set up the regression problem.
    SimpleMatrix B = model.getBetas();
    Pair<SimpleMatrix, FullTensor> moments = SpectralExperts.computeExactMoments( model );
    SimpleMatrix Pairs = moments.getValue0();
    FullTensor Triples = moments.getValue1();

    // Compute X's 
    SimpleMatrix X = RandomFactory.rand( N, D );

    // Compute y2, y3's
    SimpleMatrix y2 = MatrixFactory.zeros(N);
    SimpleMatrix y3 = MatrixFactory.zeros(N);
    for( int n = 0; n < N; n++ ) {
      DenseMatrix64F Xn = MatrixOps.row(X.getMatrix(), n );
      y2.set( n, MatrixOps.xMy( Xn, Pairs.getMatrix(), Xn ) );
      y3.set( n, Triples.project3( Xn, Xn, Xn ) );
    }

    // Regress to get Pairs_ and Triples_
    SimpleMatrix Pairs_ = algo.recoverPairsByRidgeRegression( y2, X );
    FullTensor Triples_ = algo.recoverTriplesByRegression( y3, X );

    System.out.println( algo.analysis.Pairs );
    System.out.println( Pairs_ );
    algo.analysis.reportPairs(Pairs_);
    algo.analysis.reportTriples(Triples_);

    Assert.assertTrue( algo.analysis.PairsErr < 1e-2 );
    Assert.assertTrue( algo.analysis.TriplesErr < 1e-1 );
  }

  public void arbitraryRegressionTest(SpectralExperts algo, MixtureOfExperts.GenerationOptions options) throws NumericalException {
    LogInfo.writeToStdout = true;
    LogInfo.init();

    MixtureOfExperts model = MixtureOfExperts.generate(options);
    int K = model.getK(); int D = model.getD();
    algo.K = K;
    algo.enableAnalysis(model);
    if (algo.adjustReg) {
      algo.traceReg2 /= Math.sqrt(N); //N; //Math.pow(N, 1.0/3);
      algo.traceReg3 /= Math.sqrt(N); //N; //Math.pow(N, 1.0/3);
    }

    // Set up the regression problem.
    SimpleMatrix B = model.getBetas();
    Pair<SimpleMatrix, FullTensor> moments = SpectralExperts.computeExactMoments( model );
    SimpleMatrix Pairs = moments.getValue0();
    FullTensor Triples = moments.getValue1();

    // Compute X's 
    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample((int) N);
    SimpleMatrix X = yX.getValue1();

    // Compute y2, y3's
    SimpleMatrix y2 = MatrixFactory.zeros((int) N);
    SimpleMatrix y3 = MatrixFactory.zeros((int) N);
    for( int n = 0; n < N; n++ ) {
      DenseMatrix64F Xn = MatrixOps.row(X.getMatrix(), n );
      y2.set( n, MatrixOps.xMy( Xn, Pairs.getMatrix(), Xn ) );
      y3.set( n, Triples.project3( Xn, Xn, Xn ) );
    }
    y2 = y2.transpose().plus( RandomFactory.randn(1, (int)N).scale(model.sigma2)) ;
    y3 = y3.transpose().plus( RandomFactory.randn(1, (int)N).scale(model.sigma2)) ;

    // Regress to get Pairs_ and Triples_
    MatrixOps.printSize(y2);
    MatrixOps.printSize(y3);
    MatrixOps.printSize(X);
    SimpleMatrix Pairs_ = algo.recoverPairsByRidgeRegression( y2, X );
    FullTensor Triples_ = algo.recoverTriplesByRegression( y3, X );

    System.out.println( algo.analysis.Pairs );
    System.out.println( Pairs_ );
    algo.analysis.reportPairs(Pairs_);
    algo.analysis.reportTriples(Triples_);

    System.out.printf( "%.4f %.4f\n", algo.analysis.PairsErr, algo.analysis.TriplesErr );
  }

  public void testMomentRunner(MixtureOfExperts model, int N, double reg) {
    int K = model.getK(); int D = model.getD();
    SpectralExperts algo = new SpectralExperts();
    algo.K = K;

    algo.enableAnalysis(model);

    // Compute the empirical moments
    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample(N);
    SimpleMatrix y = yX.getValue0();
    SimpleMatrix X = yX.getValue1();

    SimpleMatrix avgBetas = algo.recoverMeans(y, X);
    SimpleMatrix Pairs_ = algo.recoverPairs( y, X);
    System.out.println( algo.analysis.Pairs );
    System.out.println( Pairs_ );
    algo.analysis.reportPairs(Pairs_);
    FullTensor Triples_ = algo.recoverTriples(y, X, avgBetas);
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
  public void arbitraryMomentsTest(SpectralExperts algo, MixtureOfExperts.GenerationOptions options) throws NumericalException {
    LogInfo.writeToStdout = true;
    LogInfo.init();

    MixtureOfExperts model = MixtureOfExperts.generate(options);
    int K = model.getK(); int D = model.getD();
    algo.K = model.getK();
    algo.enableAnalysis(model);
    if (algo.adjustReg) {
      algo.traceReg2 /= Math.sqrt(N); //N; //Math.pow(N, 1.0/3);
      algo.traceReg3 /= Math.sqrt(N); //N; //Math.pow(N, 1.0/3);
    }


    // Compute the empirical moments
    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample((int)N);
    SimpleMatrix y = yX.getValue0();
    SimpleMatrix X = yX.getValue1();
    algo.analysis.checkDataSanity(y, X);

    SimpleMatrix Pairs;
    FullTensor Triples;

    if( algo.useMatlab ) {
      Pair<SimpleMatrix,FullTensor> moments = algo.recoverMomentsByMatlab( y, X );
      Pairs = moments.getValue0();
      if( algo.analysis != null ) algo.analysis.reportPairs(Pairs);

      Triples = moments.getValue1();
      if( algo.analysis != null ) algo.analysis.reportTriples(Triples);
    } else {
      // Recover the first moment
      SimpleMatrix avgBetas = algo.recoverMeans(y, X);
      if( algo.analysis != null ) algo.analysis.reportAvg(avgBetas);

      // Recover Pairs and Triples moments by linear regression
      Pairs = algo.recoverPairs( y, X );
      if( algo.analysis != null ) algo.analysis.reportPairs(Pairs);

      Triples = algo.recoverTriples(y, X, avgBetas);
      if( algo.analysis != null ) algo.analysis.reportTriples(Triples);
    }

    System.out.printf( "%.4f %.4f\n", algo.analysis.PairsErr, algo.analysis.TriplesErr );
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
  public void arbitraryRecoveryTest(SpectralExperts algo, MixtureOfExperts.GenerationOptions options) throws NumericalException, RecoveryFailure {
    LogInfo.writeToStdout = true;
    LogInfo.init();

    MixtureOfExperts model = MixtureOfExperts.generate(options);
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
  @Option( gloss = "Test the Regression?" )
  public boolean testRegression = false;
  @Option( gloss = "Test the moments?" )
  public boolean testMoments = false;

  public static void main( String[] args ) throws RecoveryFailure, NumericalException {
    SpectralExpertsTest test = new SpectralExpertsTest();
    SpectralExperts algo = new SpectralExperts();

    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    OptionsParser parser = new OptionsParser( test, algo, options );

    if( parser.parse( args ) ) {
      if( test.testRegression )
        test.arbitraryRegressionTest(algo, options);
      else if( test.testMoments )
        test.arbitraryMomentsTest(algo, options);
      else
        test.arbitraryRecoveryTest(algo, options);
    }
  }


}
