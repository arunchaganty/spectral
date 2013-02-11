package learning.optimization;

import fig.basic.LogInfo;
import org.ejml.data.DenseMatrix64F;
import learning.optimization.GradientDescentSolver;
import org.ejml.ops.CommonOps;

/**
 * Represents a problem that is solved using proximal gradient descent
 */
public class ProximalGradientSolver extends GradientDescentSolver {

  public static interface ProximalOptimizable extends GradientDescentSolver.Optimizable {
    /**
     * Compute the gradient at $x$ and store in $gradient$
     * @param x
     */
    public void projectProximal( DenseMatrix64F x );
  }

  public DenseMatrix64F optimize( ProximalOptimizable problem, DenseMatrix64F initial, int maxIters, double eps, LearningRate rate ) {
    LogInfo.begin_track("proximal-optimize");

    double lhood = problem.loss(initial);
    DenseMatrix64F state = initial.copy();
    DenseMatrix64F gradient = new DenseMatrix64F(state.numRows, state.numCols);

    for(int i = 0; i < maxIters; i++) {
      // Make a gradient step
      problem.gradient(state, gradient);
      CommonOps.addEquals(state, -rate.getRate(i), gradient);
      // Make the proximal step
      problem.projectProximal(state);

      // Compute lhood
      double lhood_ = problem.loss(state);
      LogInfo.logs("Iteration " + i + ": " + lhood_ );

      // Check convergence
      if(Math.abs(lhood - lhood_) < eps) {
        LogInfo.logsForce("Converged.");
        break;
      }
      lhood = lhood_;
    }
    LogInfo.end_track("proximal-optimize");

    return state;
  }
  public DenseMatrix64F optimize( ProximalOptimizable problem, DenseMatrix64F initial, LearningRate rate, int maxIters) {
    return optimize(problem, initial, rate, maxIters, 1e-7);
  }
  public DenseMatrix64F optimize( ProximalOptimizable problem, DenseMatrix64F initial, LearningRate rate) {
    return optimize(problem, initial, rate, 1000);
  }
  public DenseMatrix64F optimize( ProximalOptimizable problem, DenseMatrix64F initial) {
    return optimize(problem, initial, new LearningRate(LearningRate.Type.BY_SQRT_T, 0.9));
  }

}
