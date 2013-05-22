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
public class SpectralMomentsTests {

  @Before
  public void setup(){
    LogInfo.writeToStdout = false;
    LogInfo.init();
  }

  public void arbitraryOptimizationMatchTest(SpectralExperts algo, MixtureOfExperts.GenerationOptions options) throws NumericalException {
    LogInfo.writeToStdout = true;
    LogInfo.init();

    MixtureOfExperts model = MixtureOfExperts.generate(options);
    int K = model.getK(); int D = model.getD();
    algo.K = K;
    algo.enableAnalysis(model);
    if( algo.ridgeReg3 < 0 ) algo.ridgeReg3 = algo.ridgeReg2 * 10;
    if( algo.traceReg3 < 0 ) algo.traceReg3 = algo.traceReg2 * 10;
    if (algo.adjustReg) {
      algo.traceReg2 /= Math.sqrt(N); //N; //Math.pow(N, 1.0/3);
      algo.traceReg3 /= Math.sqrt(N); //N; //Math.pow(N, 1.0/3);
    }

    Pair<SimpleMatrix, FullTensor> moments = SpectralExperts.computeExactMoments( model );
    SimpleMatrix Pairs = moments.getValue0();
    FullTensor Triples = moments.getValue1();

    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample((int) N);
    SimpleMatrix y = yX.getValue0();
    SimpleMatrix X = yX.getValue1();

    SimpleMatrix PairsM, PairsR;
    FullTensor TriplesM, TriplesR;

    // Do regression with CVX
    {
      Pair<SimpleMatrix,FullTensor> momentsM = algo.recoverMomentsByMatlab( y, X );
      PairsM = momentsM.getValue0();
      TriplesM = momentsM.getValue1();
    }
    // Do regression with regression
    {
      SimpleMatrix avgBetas = algo.recoverMeans(y, X);
      PairsR = algo.recoverPairs( y, X );
      TriplesR = algo.recoverTriples(y, X, avgBetas);
    }

    System.out.printf( "%.4f %.4f %.4f %.4f %.4f %.4f\n", 
      MatrixOps.diff(PairsR, PairsM ), MatrixOps.diff(TriplesR, TriplesM ),
      MatrixOps.diff(PairsR, Pairs ), MatrixOps.diff(TriplesR, Triples ),
      MatrixOps.diff(Pairs, PairsM ), MatrixOps.diff(Triples, TriplesM )
    );
        
  }

  @Option( gloss = "Number of samples" )
  public double N = 1e4;
	@Option(gloss = "Random seed")
	public int testSeed = 0;

  public static void main( String[] args ) throws RecoveryFailure, NumericalException {
    SpectralMomentsTests test = new SpectralMomentsTests();
    SpectralExperts algo = new SpectralExperts();

    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    OptionsParser parser = new OptionsParser( test, algo, options );

    if( parser.parse( args ) ) {
      RandomFactory.setSeed( test.testSeed );
      test.arbitraryOptimizationMatchTest(algo, options);
    }
  }

}
