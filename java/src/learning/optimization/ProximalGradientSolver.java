package learning.optimization;

import fig.basic.LogInfo;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

/**
 * Represents a problem that is solved using proximal gradient descent
 */
public class ProximalGradientSolver {

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


  public static interface ProximalOptimizable {
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
    /**
     * Compute the gradient at $x$ and store in $gradient$
     * @param x
     */
    public void projectProximal( DenseMatrix64F x );
  }

  public DenseMatrix64F optimize( ProximalOptimizable problem, DenseMatrix64F initial, LearningRate rate, int maxIters, double eps ) {
    LogInfo.begin_track("proximal-optimize");

    double loss = problem.loss(initial);
    double min_loss = Double.POSITIVE_INFINITY;
    DenseMatrix64F state = initial.copy();
    DenseMatrix64F gradient = new DenseMatrix64F(state.numRows, state.numCols);

    for(int i = 0; i < maxIters; i++) {
      // Make a gradient step
      problem.gradient(state, gradient);
      CommonOps.addEquals(state, -rate.getRate(i), gradient);
      // Make the proximal step
      problem.projectProximal(state);

      // Compute loss
      double loss_ = problem.loss(state);
      LogInfo.logs("Iteration " + i + ": " + loss_ );

      // Check convergence
      if( ( Math.abs( loss - loss_ ) < eps ) || ((loss_ - min_loss) > 1e-2) ) { // Limit the amount you can go backwards
        LogInfo.logsForce("Converged.");
        break;
      }
      // Update min loss
      min_loss = ( loss_ < min_loss ) ? loss_ : min_loss;
      loss = loss_;
    }
    LogInfo.end_track("proximal-optimize");

    return state;
  }
  public DenseMatrix64F optimize( ProximalOptimizable problem, DenseMatrix64F initial, LearningRate rate, int maxIters) {
    return optimize(problem, initial, rate, maxIters, 1e-3);
  }
  public DenseMatrix64F optimize( ProximalOptimizable problem, DenseMatrix64F initial, LearningRate rate) {
    return optimize(problem, initial, rate, 1000);
  }
  public DenseMatrix64F optimize( ProximalOptimizable problem, DenseMatrix64F initial, int maxIters) {
    return optimize(problem, initial, new LearningRate(LearningRate.Type.BY_SQRT_T, 1.0),  maxIters);
  }
  public DenseMatrix64F optimize( ProximalOptimizable problem, DenseMatrix64F initial) {
    return optimize(problem, initial, new LearningRate(LearningRate.Type.BY_SQRT_T, 1.0));
  }

}
