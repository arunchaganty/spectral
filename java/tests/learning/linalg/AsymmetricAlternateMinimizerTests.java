package learning.linalg;

import fig.basic.LogInfo;
import junit.framework.Assert;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.junit.Test;

import java.util.Random;

/**
 * Test the tensor factorization methods
 */
  public class AsymmetricAlternateMinimizerTests {

  Random testRandom = new Random(3);

  /**
   * Test a single step of the eigen decomposition
   */
  @Test
  public void eigendecompositionStep() {
    AsymmetricAlternateMinimizer minimizer = new AsymmetricAlternateMinimizer();
    {
      // Construct a tensor to have just one principal direction
      SimpleMatrix x = MatrixFactory.fromVector( new double[] {1.0, 0.0, 0.0});
      FullTensor T = FullTensor.fromUnitVector(x);

      Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix> it = minimizer.initRandom(T);
      for(int i = 0; i < 100; i++) {
        it = minimizer.factorizationStep(T, it);
      }
      double value = it.getValue0();
      SimpleMatrix y = it.getValue1();
      Assert.assertEquals(value, 1.0);
      Assert.assertTrue(MatrixOps.allclose(x, y));

    }

    {
      // Construct a tensor with two principal directions
      SimpleMatrix w = new SimpleMatrix( new double[][] {{ 0.1, 0.8, 0.7 }} );
      SimpleMatrix X = new SimpleMatrix( new double[][] {
              {1.0, 0.0, 0.0},
              {0.0, 1.0, 0.0},
              {0.0, 0.0, 1.0}}
      );
      FullTensor T = FullTensor.fromDecomposition(w, X);

      Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix> it = minimizer.initRandom(T);
      for(int i = 0; i < 100; i++) {
        it = minimizer.factorizationStep(T, it);
      }
//      LogInfo.log(it);
      double value = it.getValue0();
      SimpleMatrix y = it.getValue1();
      Assert.assertEquals(value, w.get(1));
      Assert.assertTrue(
              MatrixOps.allclose(MatrixOps.col(X,1), y) ||
              MatrixOps.allclose(MatrixOps.col(X,1), y.scale(-1)) );
    }
  }

  public void eigendecompositionTest(SimpleMatrix w, SimpleMatrix X1, SimpleMatrix X2, SimpleMatrix X3) {
    FullTensor T = FullTensor.fromDecomposition(w, X1, X2, X3);

    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> results =
            new AsymmetricAlternateMinimizer().asymmetricFactorize(T, w.getNumElements());

    SimpleMatrix w_ = results.getValue0();
    w_ = MatrixOps.alignMatrixWithSigns(w_, w, true);
    SimpleMatrix X1_ = results.getValue1();
    X1_ = MatrixOps.alignMatrixWithSigns(X1_, X1, true);
    SimpleMatrix X2_ = results.getValue2();
    X2_ = MatrixOps.alignMatrixWithSigns(X2_, X2, true);
    SimpleMatrix X3_ = results.getValue3();
    X3_ = MatrixOps.alignMatrixWithSigns(X3_, X3, true);

    System.out.println( w );
    System.out.println( w_ );
    System.out.println( X1 );
    System.out.println( X1_ );
    System.out.println( X2 );
    System.out.println( X2_ );
    System.out.println( X3 );
    System.out.println( X3_ );

    System.out.println( "w: " + MatrixOps.norm(w.minus(w_)));
    System.out.println( "X1: " + MatrixOps.norm(X1.minus(X1_)));
    System.out.println( "X2: " + MatrixOps.norm(X2.minus(X2_)));
    System.out.println( "X3: " + MatrixOps.norm(X3.minus(X3_)));
    Assert.assertTrue(MatrixOps.allclose(w, w_));
    Assert.assertTrue(MatrixOps.allclose(X1, X1_));
    Assert.assertTrue( MatrixOps.allclose(X2, X2_));
    Assert.assertTrue( MatrixOps.allclose(X3, X3_));
  }

  @Test
  public void eigendecompositionIdentity() {
    SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
    SimpleMatrix X = new SimpleMatrix( new double[][] {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}}
    );
    eigendecompositionTest(w, X, X, X);
  }

  @Test
  public void eigendecompositionOrthogonal() {
    for(int i = 0; i < 5; i++) {
      SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
      SimpleMatrix X = RandomFactory.orthogonal(3);
      eigendecompositionTest(w, X, X, X);
    }
  }

  @Test
  public void eigendecompositionAssymmetricOrthogonal() {
    for(int i = 0; i < 5; i++) {
      SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
      SimpleMatrix X1 = RandomFactory.orthogonal(3);
      SimpleMatrix X2 = RandomFactory.orthogonal(3);
      SimpleMatrix X3 = RandomFactory.orthogonal(3);
      eigendecompositionTest(w, X1, X2, X3);
    }
  }

  @Test
  public void eigendecompositionAssymmetricOrthogonalCuboid() {
    for(int i = 0; i < 5; i++) {
      SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
      SimpleMatrix X1 = RandomFactory.orthogonal(3);
      SimpleMatrix X2 = RandomFactory.orthogonal(4).extractMatrix(0, 4, 0, 3);
      SimpleMatrix X3 = RandomFactory.orthogonal(5).extractMatrix(0, 5, 0, 3);;
      eigendecompositionTest(w, X1, X2, X3);
    }
  }

  @Test
  public void eigendecompositionAssymmetricRandom() {
    for(int i = 0; i < 5; i++) {
      SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
      SimpleMatrix X1 = RandomFactory.randn(testRandom, 3, 3);
      SimpleMatrix X2 = RandomFactory.randn(testRandom, 3, 3);
      SimpleMatrix X3 = RandomFactory.randn(testRandom, 3, 3);
      eigendecompositionTest(w, X1, X2, X3);
    }
  }

  @Test
  public void eigendecompositionAssymmetricRandomCuboid() {
    for(int i = 0; i < 5; i++) {
      SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
      SimpleMatrix X1 = RandomFactory.randn(testRandom, 3, 3);
      SimpleMatrix X2 = RandomFactory.randn(testRandom, 4, 3);
      SimpleMatrix X3 = RandomFactory.randn(testRandom, 5, 3);
      eigendecompositionTest(w, X1, X2, X3);
    }
  }


}
