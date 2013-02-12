package learning.optimization;

import learning.linalg.MatrixOps;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Triplet;

/**
 * Recover the low rank matrix M, with samples y = x M x.
 */
public class PhaseRecovery implements ProximalGradientSolver.ProximalOptimizable {
  DenseMatrix64F y;
  DenseMatrix64F X;
  double reg;

  public PhaseRecovery(final SimpleMatrix y, final SimpleMatrix X, final double reg) {
    this.y = y.getMatrix();
    this.X = X.getMatrix();
    this.reg = reg;
  }


  @Override
  public void gradient(DenseMatrix64F M, DenseMatrix64F gradient) {
    int N = X.numRows;
    int D = X.numCols;

    // Reset gradient
    CommonOps.fill(gradient, 0.0);

    DenseMatrix64F x = new DenseMatrix64F(1, D);
    DenseMatrix64F dGrad = new DenseMatrix64F(D,D);

    // Set gradient to be the mean (error * x)
    for( int n = 0; n < N; n++ ) {
      MatrixOps.row(X, n, x);

      double err = (MatrixOps.xMy(x, M, x) - y.get(n));
      // Compute average gradient increment = (err * x - gradient)/n+1
      MatrixOps.outer(x, x, dGrad);

      // Update gradient
      MatrixOps.incrementalAverageUpdate( err, n, dGrad, gradient );
    }
  }

  /**
   * The proximal step corresponds to soft thresholding the singular values
   * @param M - current estimate of M
   */
  @Override
  public void projectProximal( DenseMatrix64F M ) {
    int D = M.numRows;

    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> USV = MatrixOps.svdk(SimpleMatrix.wrap(M));
    SimpleMatrix U = USV.getValue0();
    SimpleMatrix S = USV.getValue1();
    SimpleMatrix V = USV.getValue2();

    // Soft threshold the singular values
    // NOTE: Using S.numRows() because the rank of the matrix could have decreased.
    for(int d = 0; d < S.numRows(); d++) {
      double v = S.get(d,d);
      v -= reg;
      S.set(d, d, (v > 0) ? v :  0 );
    }

    M.set( U.mult( S ).mult(V.transpose()).getMatrix() );
  }

  @Override
  public double loss(DenseMatrix64F M) {
    int N = X.numRows;
    int D = X.numCols;

    DenseMatrix64F x = new DenseMatrix64F(1, D);

    DenseMatrix64F y_ = new DenseMatrix64F(N, 1);
    MatrixOps.quadraticForm( X, M, y_ );

    // Set gradient to be the mean (error * x)
    double err = 0.0;
    for( int n = 0; n < N; n++ ) {
      MatrixOps.row(X, n, x);
      double yn = y.get(n);
      double pn = MatrixOps.xMy(x, M, x);
      double err_ = (pn - yn);

      err += err_ * err_;
    }
    double trace = MatrixOps.svdk(M).getValue1().elementSum();
    err = 0.5 * err + reg * trace;

    return err;
  }
}
