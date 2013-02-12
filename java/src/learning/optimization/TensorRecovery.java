package learning.optimization;

import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Triplet;

/**
 * Recover the low rank tensor.
 */
public class TensorRecovery implements ProximalGradientSolver.ProximalOptimizable {
  DenseMatrix64F y;
  DenseMatrix64F X;
  double reg;

  FullTensor T;

  public TensorRecovery(final SimpleMatrix y, final SimpleMatrix X, final double reg) {
    this.y = y.getMatrix();

    this.X = X.getMatrix();
    // Dividing by 3 because that's the number of modes/ways in the tensor
    this.reg = reg/3;

    // Some preprocessing steps

    // Pre-allocate a tensor that we will use again and again.
    int D = X.numCols();
    T = new FullTensor(D, D, D);
  }

  /**
   * Note; we are going to store the gradient as the mode-1 unfolding
   * so that we can just use the matrix infrastructure we already have
   * @param M
   * @param gradient
   */
  @Override
  public void gradient(DenseMatrix64F M, DenseMatrix64F gradient) {
    int N = X.numRows;
    int D = X.numCols;

    // Reconstruct the tensor T
    FullTensor.fold(0, M, T);

    // Reset gradient
    CommonOps.fill(gradient, 0.0);


    DenseMatrix64F x = new DenseMatrix64F(1, D);
    DenseMatrix64F dGrad = new DenseMatrix64F(D,D*D); // D, D^2 are the dimensions of the unfolding

    // Set gradient to be the mean (error * x)
    for( int n = 0; n < N; n++ ) {
      MatrixOps.row(X, n, x);

      // Compute the gradient increment
      double err = (T.project3(x,x,x) - y.get(n));
      FullTensor.unfoldUnit(0, x, dGrad);

      // Update the gradient
      MatrixOps.incrementalAverageUpdate(err, n, dGrad, gradient);
    }
  }

  /**
   * The proximal step corresponds to soft thresholding the singular values
   * @param M - current estimate of M
   */
  @Override
  public void projectProximal( DenseMatrix64F M ) {
    // Reconstruct the tensor T
    FullTensor.fold(0, M, T);
    int D = T.getDim(0);

    // Regularize along each mode
    for( int i = 0; i < 3; i++ ) {
      T.unfold(i, M);
      Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> USV = MatrixOps.svdk(M);
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
      // Refold
      FullTensor.fold(0, M, T);
    }
    T.unfold(0, M);
  }

  // Computes the sum of the trace norms along every unfolding
  public double traceNorm(FullTensor T) {
    // TODO: Is there a better way of doing this?
    int D = T.getDim(0);
    DenseMatrix64F unfolding = new DenseMatrix64F(D, D*D);

    double norm = 0.0;

    for( int i = 0; i < 3; i++ ) {
      T.unfold(i, unfolding);
      norm += MatrixOps.svdk(unfolding).getValue1().elementSum();
    }

    return norm;
  }


  @Override
  public double loss(DenseMatrix64F M) {
    int N = X.numRows;
    int D = X.numCols;

    FullTensor.fold(0, M, T);

    DenseMatrix64F x = new DenseMatrix64F(1, D);
    // Set gradient to be the mean (error * x)
    double err = 0.0;
    for( int n = 0; n < N; n++ ) {
      MatrixOps.row(X, n, x);

      double err_ = (T.project3(x,x,x) - y.get(n));
      err += err_ * err_;
    }
    // Really this should be the sum of the unfoldings
    double trace = traceNorm(T);
    err = 0.5 * err + reg * trace;

    return err;
  }
}
