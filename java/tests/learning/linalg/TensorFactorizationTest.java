package learning.linalg;

import junit.framework.Assert;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.junit.Test;

/**
 * Test the tensor factorization methods
 */
public class TensorFactorizationTest {

  /**
   * Test a single step of the eigen decomposition
   */
  @Test
  public void eigendecompositionStep() {
    // Construct a tensor to have just one principal direction
    SimpleMatrix x = MatrixFactory.fromVector( new double[] {1.0, 0.0, 0.0});

    FullTensor T = FullTensor.fromUnitVector(x);
    Pair<Double, SimpleMatrix> pair = TensorFactorization.eigendecomposeStep(T, 10, 1);
    double value = pair.getValue0();
    SimpleMatrix y = pair.getValue1();
    Assert.assertEquals(value, 1.0);
    Assert.assertTrue(MatrixOps.allclose(x, y));

    // Construct a tensor with two principal directions
    SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
    SimpleMatrix X = new SimpleMatrix( new double[][] {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}}
            );

    T = FullTensor.fromDecomposition(w, X);
    pair = TensorFactorization.eigendecomposeStep(T, 10, 10);
    value = pair.getValue0();
    y = pair.getValue1();
    Assert.assertEquals(value, w.get(0));
    Assert.assertTrue(MatrixOps.allclose(MatrixOps.col(X,0), y));
  }

  public void eigendecompositionTest(SimpleMatrix w, SimpleMatrix X, boolean whiten) {
    FullTensor T = FullTensor.fromDecomposition(w, X);
    SimpleMatrix P = X.mult(MatrixFactory.diag(w)).mult(X.transpose());
    Pair<SimpleMatrix, SimpleMatrix> pair = (whiten)
            ? TensorFactorization.eigendecompose(T, P)
            : TensorFactorization.eigendecompose(T);

    SimpleMatrix w_ = pair.getValue0();
    w_ = MatrixOps.alignMatrix( w_, w, true );
    SimpleMatrix X_ = pair.getValue1();
    X_ = MatrixOps.alignMatrix( X_, X, true );

    System.out.println( w );
    System.out.println( w_ );
    System.out.println( X );
    System.out.println( X_ );

    System.out.println( "w: " + MatrixOps.norm(w.minus(w_)));
    System.out.println( "X: " + MatrixOps.norm(X.minus(X_)));
    Assert.assertTrue( MatrixOps.allclose(w, w_));
    Assert.assertTrue( MatrixOps.allclose(X, X_));
  }

  @Test
  public void eigendecompositionIdentity() {
    SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
    SimpleMatrix X = new SimpleMatrix( new double[][] {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}}
    );
    eigendecompositionTest(w,X, false);
  }

  @Test
  public void eigendecompositionOrthogonal() {
    for(int i = 0; i < 5; i++) {
      SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
      SimpleMatrix X = RandomFactory.orthogonal(3);
      eigendecompositionTest(w, X, false);
    }
  }

  @Test
  public void eigendecompositionOrthogonalWithWhitening() {
    for(int i = 0; i < 5; i++) {
      SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
      SimpleMatrix X = RandomFactory.orthogonal(3);
      eigendecompositionTest(w, X, true);
    }
  }

  @Test
  public void eigendecompositionRandom() {
    for(int i = 0; i < 5; i++) {
      SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
      SimpleMatrix X = RandomFactory.rand(3, 3);
      eigendecompositionTest(w, X, true);
    }
  }

  @Test
  public void eigendecompositionRandomNonSquare() {
    for(int i = 0; i < 5; i++) {
      SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
      SimpleMatrix X = RandomFactory.rand(5, 3);
      eigendecompositionTest(w, X, true);
    }
  }

}
