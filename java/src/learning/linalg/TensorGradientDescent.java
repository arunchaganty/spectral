package learning.linalg;

import fig.basic.*;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import java.util.Random;

/**
 * Solves the power method step using gradient descent
 */
public class TensorGradientDescent extends DeflatingTensorFactorizationAlgorithm {
  @Option(gloss="Number of attempts in order to find the real maxima")
  public int attempts = 100;
  @Option(gloss="Number of iterations to run the power method")
  public int iters = 50;
  @Option(gloss="Random number generator for tensor method and random projections")
  Random rnd = new Random(23);
  @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
  @OptionSet(name="backtrack") public BacktrackingLineSearch.Options backtrack = new BacktrackingLineSearch.Options();

  /**
   * f(x) = \| T(x,x,x) - x^{3} \|.
   */
  public static class TensorFunctionState implements Maximizer.FunctionState {
    double[] point;
    FullTensor T;
    double value;
    double[] gradient;
    boolean objectiveValid = false;
    boolean gradientValid = false;

    TensorFunctionState(FullTensor T) {
      this.T = T;
      assert( T.D1 == T.D2 && T.D2 == T.D3 );
      point = new double[T.D1];
      gradient = new double[T.D1];
    }

    @Override
    public double[] point() {
      return point;
    }

    @Override
    public double value() {
      if( !objectiveValid ) {
        value = -(Math.pow(MatrixOps.norm(T),2.0) + Math.pow(MatrixOps.norm(point),6.0) - 2 * T.project3(point,point,point));
        objectiveValid = true;
      }
      return value;
    }

    @Override
    public double[] gradient() {
      if( !gradientValid ) {
        for( int i = 0; i < T.D1; i++ ) {
          gradient[i] = -(- 6 * T.project3(point, point, MatrixFactory.unitVector(T.D1, i)) +
                  6 * Math.pow(MatrixOps.norm(point),4.0) * point[i]);
        }
        gradientValid = true;
      }
      return gradient;
    }

    @Override
    public void invalidate() {
      objectiveValid = gradientValid = false;
    }
  }

  protected Pair<Double,SimpleMatrix> eigendecomposeStep( FullTensor T ) {
    LogInfo.begin_track("GradientPowerMethod:step");
    Maximizer max = new LBFGSMaximizer(backtrack, lbfgs);
    TensorFunctionState state = new TensorFunctionState(T);

    double maxEigenvalue = Double.NEGATIVE_INFINITY;
    SimpleMatrix maxEigenvector = null;

    for( int attempt = 0; attempt < attempts; attempt++ ) {
      if( attempt % 10 == 0 )
        LogInfo.logs("Attempt %d/%d", attempt, attempts);
      for( int i = 0; i < state.point.length; i++ )
        state.point[i] = rnd.nextDouble();//maxEigenvector[i]; //1.0/(1+state.point.length);

      boolean done = false;
      LogInfo.begin_track("Optimization");
      for( int i = 0; i < iters && !done; i++ ) {
        done = max.takeStep(state);
        LogInfo.logs("%s objective = %f, point = %s, gradient = %s", i, state.value(), Fmt.D(state.point()), Fmt.D(state.gradient()));
      }
      LogInfo.end_track();
      SimpleMatrix eigenvector = MatrixFactory.fromVector(state.point());
      double eigenvalue = T.project3(eigenvector, eigenvector, eigenvector);
      if( eigenvalue > maxEigenvalue ) {
        maxEigenvalue = eigenvalue;
        maxEigenvector = eigenvector;
      }
    }
    LogInfo.end_track("GradientPowerMethod:step");

    return Pair.with(maxEigenvalue, maxEigenvector);
  }
}
