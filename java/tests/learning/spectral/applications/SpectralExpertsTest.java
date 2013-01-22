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

  @Test
  public void testMoments() {
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.K = 2; options.D = 2; options.betas = "eye"; options.bias = false;

    MixtureOfExperts model = MixtureOfExperts.generate(options);
    int K = model.getK(); int D = model.getD();
    SpectralExperts algo = new SpectralExperts();

    // Compute exact moments
    Pair<SimpleMatrix, Tensor> moments = algo.computeExactMoments( model );
    SimpleMatrix Pairs = moments.getValue0();
    Tensor Triples = moments.getValue1();

    // Compute the empirical moments
    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample(10000);
    SimpleMatrix y = yX.getValue0();
    SimpleMatrix X = yX.getValue1();

    SimpleMatrix Pairs_ = algo.recoverPairs( y, X, 0.0001 );
    Tensor Triples_ = algo.recoverTriples(y, X, 0.0001);

    // Compare errors
    double err = MatrixOps.norm( Pairs.minus( Pairs_ ) );
    System.err.println( Pairs );
    System.err.println( Pairs_ );
    System.err.println( "Pairs: " + err );
    // Assert.assertTrue( err < 1e-3 );

    SimpleMatrix eta = RandomFactory.rand(D, 1);
    SimpleMatrix TriplesT = Triples.project(2, eta );
    SimpleMatrix TriplesT_ = Triples_.project(2, eta );
    System.err.println( TriplesT );
    System.err.println( TriplesT_ );

    err = MatrixOps.norm( TriplesT.minus( TriplesT_ ) );
    System.err.println( "Triples: " + err );
//
//    Assert.assertTrue( err < 1e-2 );
  }

}
