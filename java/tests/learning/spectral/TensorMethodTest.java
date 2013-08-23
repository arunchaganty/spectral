/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral;

import learning.data.ComputableMoments;
import learning.linalg.*;

import learning.models.MixtureOfGaussians;

import static learning.models.MixtureOfGaussiansTest.*;

import org.javatuples.*;

import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.basic.LogInfo;

/**
 * 
 */
public class TensorMethodTest {

  @Before 
  public void setUp() {
    LogInfo.writeToStdout = false;
    LogInfo.init();
  }
  // Actual tests

  public void testSymmetrization( MixtureOfGaussians model ) {
    int K = model.getK();
    int D = model.getD();
    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments = 
        model.computeExactMoments();

    //Triplet<Pair<SimpleMatrix, FullTensor>,
    //    Pair<SimpleMatrix, FullTensor>,
    //    Pair<SimpleMatrix, FullTensor>>
    //    symmetricMoments = model.computeSymmetricMoments();
    Pair<SimpleMatrix,FullTensor> symmetricMoments = model.computeSymmetricMoments().getValue2();

    TensorMethod algo = new TensorMethod();
    Pair<SimpleMatrix, FullTensor> symmetrizedMoments = TensorMethod.symmetrizeViews( 
        K,
        moments.getValue0(), // M12
        moments.getValue1(), // M13
        moments.getValue2(), // M23
        moments.getValue3() ); // M123
    // Test that these are symmetric indeed.
    SimpleMatrix Pairs = symmetrizedMoments.getValue0();
    FullTensor Triples = symmetrizedMoments.getValue1();

    // Property tests
    Assert.assertTrue( Pairs.numRows() == D ) ; 
    Assert.assertTrue( Pairs.numCols() == D ) ; 
    Assert.assertTrue( Triples.D1 == D ) ; 
    Assert.assertTrue( Triples.D2 == D ) ; 
    Assert.assertTrue( Triples.D3 == D ) ; 

    Assert.assertTrue( MatrixOps.isSymmetric( Pairs ) );
    Assert.assertTrue( MatrixOps.isSymmetric( Triples ) );

    // Equality Tests
    Assert.assertTrue( MatrixOps.allclose( symmetricMoments.getValue0(), Pairs ) );
    Assert.assertTrue( MatrixOps.allclose( symmetricMoments.getValue1(), Triples ) );
  }

  @Test
  public void testSymmetrizationSmallEye() { testSymmetrization( generateSmallEye() ); }
  @Test
  public void testSymmetrizationSmallRandom() { testSymmetrization( generateSmallRandom() ); }
  @Test
  public void testSymmetrizationMediumEye() {  testSymmetrization( generateMediumEye() ); }
  @Test
  public void testSymmetrizationMediumRandom() {  testSymmetrization( generateMediumRandom() ); }

  public void testExactSymmetricRunner( MixtureOfGaussians model ) {
    int K = model.getK();
    int D = model.getD();
    int V = model.getV();

    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments = 
        model.computeExactMoments();
    SimpleMatrix Pairs = moments.getValue0();
    FullTensor Triples = moments.getValue3();
    TensorMethod algo = new TensorMethod();
    Pair<SimpleMatrix, SimpleMatrix> params = algo.recoverParameters( K, Pairs, Triples  );
    SimpleMatrix weights_ = params.getValue0();
    SimpleMatrix M_ = params.getValue1();

    // Properties

    Assert.assertTrue( weights_.numRows() == 1 );
    Assert.assertTrue( weights_.numCols() == K );

    Assert.assertTrue( M_.numRows() == D );
    Assert.assertTrue( M_.numCols() == K );

    // Exact values
    SimpleMatrix weights = model.getWeights();
    SimpleMatrix M = model.getMeans()[0];

    M_ = MatrixOps.alignMatrix( M_, M, true ); 

    Assert.assertTrue( MatrixOps.allclose( weights, weights_) );
    Assert.assertTrue( MatrixOps.allclose( M, M_) );
  }

  @Test
  public void testExactSmallSymmetric() { testExactSymmetricRunner( generateSmallSymmetric() ); }
  @Test
  public void testExactMediumSymmetric() { testExactSymmetricRunner( generateMediumSymmetric() ); }

  public void testExactRunner( MixtureOfGaussians model ) {
    int K = model.getK();
    int D = model.getD();
    int V = model.getV();

    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments = 
        model.computeExactMoments();

    TensorMethod algo = new TensorMethod();
    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> params = algo.recoverParameters( K, moments );
    SimpleMatrix weights_ = params.getValue0();
    SimpleMatrix M1_ = params.getValue1();
    SimpleMatrix M2_ = params.getValue2();
    SimpleMatrix M3_ = params.getValue3();

    // Properties

    Assert.assertTrue( weights_.numRows() == 1 );
    Assert.assertTrue( weights_.numCols() == K );
    Assert.assertTrue( M1_.numRows() == D );
    Assert.assertTrue( M1_.numCols() == K );
    Assert.assertTrue( M2_.numRows() == D );
    Assert.assertTrue( M2_.numRows() == D );
    Assert.assertTrue( M3_.numCols() == K );
    Assert.assertTrue( M3_.numCols() == K );

    // Exact values
    SimpleMatrix weights = model.getWeights();
    SimpleMatrix M1 = model.getMeans()[0];
    SimpleMatrix M2 = model.getMeans()[1];
    SimpleMatrix M3 = model.getMeans()[2];

    M1_ = MatrixOps.alignMatrix( M1_, M1, true ); 
    M2_ = MatrixOps.alignMatrix( M2_, M2, true ); 
    M3_ = MatrixOps.alignMatrix( M3_, M3, true ); 

    Assert.assertTrue( MatrixOps.allclose( weights, weights_) );
    Assert.assertTrue( MatrixOps.allclose( M1, M1_) );
    Assert.assertTrue( MatrixOps.allclose( M2, M2_) );
    Assert.assertTrue( MatrixOps.allclose( M3, M3_) );
  }

  @Test
  public void testExactSmallSymmetric_() { testExactRunner( generateSmallSymmetric() ); }
  @Test
  public void testExactMediumSymmetric_() { testExactRunner( generateMediumSymmetric() ); }


  public void testSampleRunner( int n, int K, int D, int V, MixtureOfGaussians model ) {
  }
  // @Test
  public void testSamples() {};

  @Option( gloss = "Number of points" )
  public double N = 1e4;
  
  public static void main( String[] args ) {
    TensorMethodTest test = new TensorMethodTest();
    MixtureOfGaussians.GenerationOptions genOptions = new MixtureOfGaussians.GenerationOptions();
    OptionsParser parser = new OptionsParser( test, genOptions );
    if( parser.parse( args ) ) {
    }
  }

  public void testProjection( MixtureOfGaussians model ) {
    int K = model.getK();
    int D = model.getD();
    ComputableMoments moments = model.computeExactMoments_();
    Pair<SimpleMatrix,FullTensor> symmetricMoments = model.computeSymmetricMoments().getValue2();

    TensorMethod algo = new TensorMethod();
    FullTensor Triples = algo.randomizedSymmetricTensorRecovery(K, moments).getValue1();

    // Test that these are symmetric indeed.

    // Property tests
    Assert.assertTrue( Triples.D1 == K ) ;
    Assert.assertTrue( Triples.D2 == K ) ;
    Assert.assertTrue( Triples.D3 == K ) ;

    Assert.assertTrue( MatrixOps.isSymmetric( Triples ) );

    // Equality Tests
    SimpleMatrix W = MatrixOps.whitener(symmetricMoments.getValue0(), K);
    FullTensor T = symmetricMoments.getValue1().rotate(W, W, W);
    FullTensor Tm = T.clone().scale(-1.0);
    // Either one is possible because of $W$
    Assert.assertTrue( MatrixOps.allclose( T, Triples ) || MatrixOps.allclose( Tm, Triples ));
  }
  @Test
  public void testProjection() { testProjection(generateSymmetricSparseEye()); }

  public void testProjectedSymmetrization( MixtureOfGaussians model ) {
    int K = model.getK();
    int D = model.getD();
    ComputableMoments moments = model.computeExactMoments_();

    Pair<SimpleMatrix,FullTensor> symmetricMoments = model.computeSymmetricMoments().getValue2();

    TensorMethod algo = new TensorMethod();
    FullTensor Triples = algo.randomizedTensorRecovery(K, moments).getValue1();

    // Test that these are symmetric indeed.

    // Property tests
    Assert.assertTrue( Triples.D1 == K ) ;
    Assert.assertTrue( Triples.D2 == K ) ;
    Assert.assertTrue( Triples.D3 == K ) ;

    Assert.assertTrue( MatrixOps.isSymmetric(Triples));

    // Equality Tests
    SimpleMatrix W = MatrixOps.whitener(symmetricMoments.getValue0(), K);
    FullTensor T = symmetricMoments.getValue1().rotate(W, W, W);
    FullTensor Tm = T.clone().scale(-1.0);
    System.out.println( Math.min( MatrixOps.maxdiff( T, Triples), MatrixOps.maxdiff( Tm, Triples) ) );
    Assert.assertTrue( MatrixOps.allclose( T, Triples, 1e-3 ) || MatrixOps.allclose( Tm, Triples, 1e-3 ));
  }

  @Test
  public void testProjectedSymmetrization() { testProjectedSymmetrization(generateUnSymmetricSparseEye()); }

  // Note: this test fails
  @Test
  public void testProjectedSymmetrizationRandom() { testProjectedSymmetrization(generateUnSymmetricSparseRandom()); }

}

