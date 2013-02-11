package learning.optimization;

import fig.basic.LogInfo;
import learning.linalg.MatrixOps;
import learning.models.MixtureOfExperts;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.junit.Assert;
import org.junit.Test;

/**
 * Test for gradient descent solver
 */
public class GradientDecsentSolverTest {

  @Test
  public void optimize() {
    // Use a single mixture to generate a regression problem to solve.
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.D = 3; options.K = 1; options.betas = "random"; options.weights = "random";
    MixtureOfExperts model = MixtureOfExperts.generate(options);
    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample(1000);
    final DenseMatrix64F y = yX.getValue0().getMatrix();
    final DenseMatrix64F X = yX.getValue1().getMatrix();
    final int N = X.numRows;
    final int D = X.numCols;

    // Define a simple problem.
    DenseMatrix64F initialState = new DenseMatrix64F(1, D);
    DenseMatrix64F betas = model.getBetas().transpose().getMatrix();

    GradientDescentSolver solver = new GradientDescentSolver();
    DenseMatrix64F betas_ = solver.optimize(new GradientDescentSolver.Optimizable() {
      @Override
      public void gradient(DenseMatrix64F betas_, DenseMatrix64F gradient) {
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
          CommonOps.addEquals( gradient, x );
        }
      }

      @Override
      public double loss(DenseMatrix64F betas_) {
        DenseMatrix64F x = new DenseMatrix64F(1, D);

        // Set gradient to be the mean (error * x)
        double err = 0.0;
        for( int n = 0; n < N; n++ ) {
          MatrixOps.row(X, n, x);

          err += ( Math.pow( MatrixOps.dot( x, betas_ ) - y.get(n), 2 ) - err)/(n+1);
        }
        return err;
      }
    }, initialState, new GradientDescentSolver.LearningRate(GradientDescentSolver.LearningRate.Type.BY_SQRT_T, 0.99), 100, 1e-6);

    System.out.println( betas_ );
    System.out.println( betas );
    LogInfo.logs( "Error: " + MatrixOps.norm( SimpleMatrix.wrap( betas ).minus( SimpleMatrix.wrap(betas_))));

    Assert.assertTrue(MatrixOps.allclose( betas, betas_, 1e-2));
  }
}
