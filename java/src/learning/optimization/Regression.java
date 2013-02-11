package learning.optimization;

import learning.linalg.MatrixOps;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;

/**
 * Regression as GD
 */
public class Regression implements ProximalGradientSolver.ProximalOptimizable {

  DenseMatrix64F X;
  DenseMatrix64F y;
  public Regression(SimpleMatrix y, SimpleMatrix X) {
    this(y.getMatrix(), X.getMatrix());
  }
  public Regression(DenseMatrix64F y, DenseMatrix64F X) {
    this.y = y; this.X = X;
  }

  @Override
  public void gradient(DenseMatrix64F betas_, DenseMatrix64F gradient) {
    int N = X.numRows; int D = X.numCols;

    // Reset gradient
    CommonOps.fill( gradient, 0.0 );

    DenseMatrix64F x = new DenseMatrix64F(1, D);

    // Set gradient to be the mean (error * x)
    for( int n = 0; n < N; n++ ) {
      MatrixOps.row(X, n, x);

      double err = (MatrixOps.dot(x, betas_) - y.get(n));
      // Compute average gradient increment = (err * x - gradient)/n+1
      CommonOps.scale(err, x);
      CommonOps.subEquals(x, gradient);
      CommonOps.scale(1.0 / (n + 1), x);

      // Increment gradient
      CommonOps.addEquals(gradient, x);
    }
  }

  @Override
  public void projectProximal(DenseMatrix64F state) {}

  @Override
  public double loss(DenseMatrix64F betas_) {
    int N = X.numRows; int D = X.numCols;

    DenseMatrix64F x = new DenseMatrix64F(1, D);

    // Set gradient to be the mean (error * x)
    double err = 0.0;
    for( int n = 0; n < N; n++ ) {
      MatrixOps.row(X, n, x);

      err += ( Math.pow( MatrixOps.dot( x, betas_ ) - y.get(n), 2 ) - err)/(n+1);
    }
    return err;
  }
}
