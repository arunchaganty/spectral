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

  @Test
  public void eigendecompositionIdentity() {
    // Construct a tensor with two principal directions
    SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
    SimpleMatrix X = new SimpleMatrix( new double[][] {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}}
    );

    FullTensor T = FullTensor.fromDecomposition(w, X);
    Pair<SimpleMatrix, SimpleMatrix> pair = TensorFactorization.eigendecompose(T);
    SimpleMatrix w_ = pair.getValue0();
    SimpleMatrix X_ = pair.getValue1();
    System.out.println( "w: " + MatrixOps.norm(w.minus(w_)));
    System.out.println( "X: " + MatrixOps.norm(X.minus(X_)));
    Assert.assertTrue( MatrixOps.allclose(w, w_));
    Assert.assertTrue( MatrixOps.allclose(X, X_));
  }

  @Test
  public void eigendecompositionIdentityWithWhitening() {
    // Construct a tensor with two principal directions
    SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
    SimpleMatrix X = new SimpleMatrix( new double[][] {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}}
    );

    FullTensor T = FullTensor.fromDecomposition(w, X);
    SimpleMatrix P = X.mult(MatrixFactory.diag(w)).mult(X.transpose());
    Pair<SimpleMatrix, SimpleMatrix> pair = TensorFactorization.eigendecompose(T, P);

    SimpleMatrix w_ = pair.getValue0();
    w_ = MatrixOps.alignMatrix( w_, w, true );
    SimpleMatrix X_ = pair.getValue1();
    X_ = MatrixOps.alignMatrix( X_, X, true );

    System.err.println( w );
    System.err.println( w_ );
    System.err.println( X );
    System.err.println( X_ );

    System.out.println( "w: " + MatrixOps.norm(w.minus(w_)));
    System.out.println( "X: " + MatrixOps.norm(X.minus(X_)));
    Assert.assertTrue( MatrixOps.allclose(w, w_));
    Assert.assertTrue( MatrixOps.allclose(X, X_));
  }

  @Test
  public void eigendecompositionOrthogonal() {
    // Construct a tensor with two principal directions
    SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
    SimpleMatrix X = RandomFactory.orthogonal(3);

    FullTensor T = FullTensor.fromDecomposition(w, X);
    Pair<SimpleMatrix, SimpleMatrix> pair = TensorFactorization.eigendecompose(T);
    SimpleMatrix w_ = pair.getValue0();
    w_ = MatrixOps.alignMatrix( w_, w, true );
    SimpleMatrix X_ = pair.getValue1();
    X_ = MatrixOps.alignMatrix( X_, X, true );

    System.err.println( w );
    System.err.println( w_ );
    System.err.println( X );
    System.err.println( X_ );

    System.err.println( "w: " + MatrixOps.norm(w.minus(w_)));
    System.err.println( "X: " + MatrixOps.norm(X.minus(X_)));

    Assert.assertTrue( MatrixOps.allclose(w, w_));
    Assert.assertTrue( MatrixOps.allclose(X, X_));
  }

  @Test
  public void eigendecompositionOrthogonalWithWhitening() {
    // Construct a tensor with two principal directions
    SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
    SimpleMatrix X = RandomFactory.orthogonal(3);

    FullTensor T = FullTensor.fromDecomposition(w, X);
    SimpleMatrix P = X.mult(MatrixFactory.diag(w)).mult(X.transpose());
    Pair<SimpleMatrix, SimpleMatrix> pair = TensorFactorization.eigendecompose(T, P);
    SimpleMatrix w_ = pair.getValue0();
    w_ = MatrixOps.alignMatrix( w_, w, true );
    SimpleMatrix X_ = pair.getValue1();
    X_ = MatrixOps.alignMatrix( X_, X, true );

    System.err.println( w );
    System.err.println( w_ );
    System.err.println( X );
    System.err.println( X_ );

    System.err.println( "w: " + MatrixOps.norm(w.minus(w_)));
    System.err.println( "X: " + MatrixOps.norm(X.minus(X_)));

    Assert.assertTrue( MatrixOps.allclose(w, w_));
    Assert.assertTrue( MatrixOps.allclose(X, X_));
  }

  @Test
  public void eigendecompositionRandom() {
    // Construct a tensor with two principal directions
    SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
    SimpleMatrix X = RandomFactory.rand(3, 3);

    FullTensor T = FullTensor.fromDecomposition(w, X);
    SimpleMatrix P = X.mult(MatrixFactory.diag(w)).mult(X.transpose());
    Pair<SimpleMatrix, SimpleMatrix> pair = TensorFactorization.eigendecompose(T, P);

    SimpleMatrix w_ = pair.getValue0();
    w_ = MatrixOps.alignMatrix( w_, w, true );
    SimpleMatrix X_ = pair.getValue1();
    X_ = MatrixOps.alignMatrix( X_, X, true );

    System.err.println( w );
    System.err.println( w_ );
    System.err.println( X );
    System.err.println( X_ );

    System.out.println( "w: " + MatrixOps.norm(w.minus(w_)));
    System.out.println( "X: " + MatrixOps.norm(X.minus(X_)));

    Assert.assertTrue( MatrixOps.allclose(w, w_));
    Assert.assertTrue( MatrixOps.allclose(X, X_));
  }
}
