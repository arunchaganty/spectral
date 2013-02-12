package learning.linalg;

import junit.framework.Assert;
import org.ejml.simple.SimpleMatrix;
import org.junit.Test;

/**
 * Tensor Tests
 */
public class FullTensorTest {

  FullTensor randomTensor(int K, int D) {
    SimpleMatrix w = RandomFactory.rand(1, K);
    // Normalize
    w = MatrixOps.normalize(w.elementMult(w));

    SimpleMatrix X = RandomFactory.rand(K, D);
    return FullTensor.fromDecomposition(w, X);
  }

  @Test
  public void rotate() {
    SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
    SimpleMatrix X = RandomFactory.rand(3, 3);
    FullTensor T = FullTensor.fromDecomposition(w, X);
    SimpleMatrix P = X.mult(MatrixFactory.diag(w)).mult(X.transpose());
    SimpleMatrix W = MatrixOps.whitener(P);

    FullTensor T1 = T.rotate(W, W, W);
    // This is just to verify that the two logics are identical.
    // Never use rotateSlow
    @SuppressWarnings("deprecation")
    FullTensor T2 = T.rotateSlow(W, W, W);

    Assert.assertTrue( MatrixOps.allclose(T1, T2) );
  }

  @Test
  public void testOrthogonal() {
    SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
    SimpleMatrix X = RandomFactory.orthogonal(3);
    FullTensor T = FullTensor.fromDecomposition(w, X);

    SimpleMatrix U1 = MatrixOps.col(X, 0);
    SimpleMatrix U2 = MatrixOps.col(X, 1);
    SimpleMatrix U3 = MatrixOps.col(X, 2);

    System.out.println( T.project3( U1, U1, U1 ) );
    System.out.println( T.project3( U2, U2, U2 ) );
    System.out.println( T.project3( U3, U3, U3 ) );
    System.out.println( T.project3( U1, U1, U2 ) );
    System.out.println( T.project3( U1, U1, U3 ) );
    System.out.println( T.project3( U2, U2, U1 ) );
    System.out.println( T.project3( U2, U2, U3 ) );
    System.out.println( T.project3( U3, U3, U1 ) );
    System.out.println( T.project3( U3, U3, U2 ) );

    Assert.assertTrue( MatrixOps.equal( T.project3( U1, U1, U1 ), w.get(0) ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U2, U2, U2 ), w.get(1) ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U3, U3, U3 ), w.get(2) ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U1, U1, U2 ), 0 ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U1, U1, U3 ), 0 ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U2, U2, U1 ), 0 ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U2, U2, U3 ), 0 ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U3, U3, U1 ), 0 ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U3, U3, U2 ), 0 ) );
  }

  @Test
  public void testWhitened() {
    SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.1, 0.8, 0.7 }} );
    SimpleMatrix X = RandomFactory.rand(3, 3);
    FullTensor T = FullTensor.fromDecomposition(w, X);
    SimpleMatrix P = X.mult(MatrixFactory.diag(w)).mult(X.transpose());
    SimpleMatrix U = P.svd().getU();
    SimpleMatrix W = MatrixOps.whitener(P);

    T = T.rotate(W, W, W);
    X = W.transpose().mult(X);
    SimpleMatrix U1 = MatrixOps.col(X, 0);
    SimpleMatrix U2 = MatrixOps.col(X, 1);
    SimpleMatrix U3 = MatrixOps.col(X, 2);


    System.out.println( T.project3( U1, U1, U1 ) );
    System.out.println( T.project3( U2, U2, U2 ) );
    System.out.println( T.project3( U3, U3, U3 ) );
    System.out.println( T.project3( U1, U1, U2 ) );
    System.out.println( T.project3( U1, U1, U3 ) );
    System.out.println( T.project3( U2, U2, U1 ) );
    System.out.println( T.project3( U2, U2, U3 ) );
    System.out.println( T.project3( U3, U3, U1 ) );
    System.out.println( T.project3( U3, U3, U2 ) );

    Assert.assertTrue( MatrixOps.equal( T.project3( U1, U1, U1 ), Math.pow( w.get(0), -2 ) ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U2, U2, U2 ), Math.pow( w.get(1), -2 ) ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U3, U3, U3 ), Math.pow( w.get(2), -2 ) ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U1, U1, U2 ), 0 ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U1, U1, U3 ), 0 ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U2, U2, U1 ), 0 ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U2, U2, U3 ), 0 ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U3, U3, U1 ), 0 ) );
    Assert.assertTrue( MatrixOps.equal( T.project3( U3, U3, U2 ), 0 ) );
  }

  @Test
  public void unfold() {
    FullTensor T = randomTensor(3, 3);
    // Test by unfolding and refolding - do we get the same thing back?
    for (int i = 0; i < 3; i++ ) {
      SimpleMatrix M = T.unfold(i);
      FullTensor T_ = FullTensor.fold(i, M, new int[] {T.D1,T.D2,T.D3});
      Assert.assertTrue( MatrixOps.allclose(T, T_ ));
    }
  }

}
