package learning.optimization;

import fig.basic.LogInfo;
import learning.linalg.MatrixOps;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

/**
 * Describes a solver for gradient descent
 */
public class GradientDescentSolver {

  /**
   * Interface for an optimizable entity
   */
  public static interface Optimizable {
    /**
     * Compute the gradient at $x$ and store in $gradient$
     * @param x
     * @param gradient
     */
    public void gradient( DenseMatrix64F x, DenseMatrix64F gradient );

    /**
     * Return the loss/log-likelihood score at a particular $x$
     * @param x
     * @return
     */
    public double loss( DenseMatrix64F x );
  }

  public static class LearningRate {
      public static enum Type {
        CONSTANT,
        BY_T,
        BY_SQRT_T
      }

      public Type type;
      public double rate;

      public LearningRate(Type type, double initialRate) {
        this.type = type;
        this.rate = initialRate;
      }

      public double getRate(double t) {
        switch(type) {
          case CONSTANT: return rate;
          case BY_T: return rate/t+1;
          case BY_SQRT_T: return rate/Math.sqrt(t + 1);
        }
        return rate;
      }
    }

  /**
   * Optimize a problem from an initial value
   * @param problem - Problem that defines a gradient
   * @param initial - Initial point for the problem
   * @param maxIters -
   * @param eps - terminate when d(loss) < eps
   */
  public DenseMatrix64F optimize( Optimizable problem, DenseMatrix64F initial, LearningRate rate, int maxIters, double eps ) {
    LogInfo.begin_track("optimize");

    double lhood = problem.loss(initial);

    DenseMatrix64F state = initial.copy();
    DenseMatrix64F gradient = new DenseMatrix64F(state.numRows, state.numCols);

    for(int i = 0; i < maxIters; i++) {
      // Make a gradient step
      problem.gradient(state, gradient);
      CommonOps.addEquals( state, -rate.getRate(i), gradient );
      // Compute lhood
      double lhood_ = problem.loss(state);
      LogInfo.logs("Iteration " + i + ": " + lhood_ + " gradient: " + MatrixOps.norm( gradient ) );

      // Check convergence
      if(Math.abs(lhood - lhood_) < eps) {
        LogInfo.logsForce("Converged.");
        break;
      }
      lhood = lhood_;
    }
    LogInfo.end_track("optimize");

    return state;
  }
  public DenseMatrix64F optimize( Optimizable problem, DenseMatrix64F initial, LearningRate rate, int maxIters) {
    return optimize(problem, initial, rate, maxIters, 1e-7);
  }
  public DenseMatrix64F optimize( Optimizable problem, DenseMatrix64F initial, LearningRate rate) {
    return optimize(problem, initial, rate, 1000);
  }
  public DenseMatrix64F optimize( Optimizable problem, DenseMatrix64F initial) {
    return optimize(problem, initial, new LearningRate(LearningRate.Type.BY_SQRT_T, 0.9));
  }


}
