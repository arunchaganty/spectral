package learning.linalg;

import fig.basic.*;
import fig.exec.Execution;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.javatuples.Quartet;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Jointly solve for a factorization for T and P:
 */
public class TensorJointGradientDescent implements  TensorFactorizationAlgorithm {

  @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
  @OptionSet(name="backtrack") public BacktrackingLineSearch.Options backtrack = new BacktrackingLineSearch.Options();;

  /**
   * f(x) = \| T(x,x,x) - x^{3} \|.
   */
  private static class SymmetricStateVectors implements Maximizer.FunctionState {
    int K;
    FullTensor T;

    private final double[] lambda;
    final double[] point; // K times D parameters
    double value;
    final double[] gradient;
    boolean objectiveValid = false;
    boolean gradientValid = false;

    SymmetricStateVectors(FullTensor T, int K, double[] lambda) {
      assert( MatrixOps.isSymmetric( T ) );
      assert( lambda.length == K );
      this.T = T;
      this.lambda = lambda;
      point = new double[K * T.D1];
      gradient = new double[K * T.D1];
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

  boolean optimize( Maximizer maximizer, Maximizer.FunctionState state, String label, int numIters ) {
    LogInfo.begin_track("optimize " + label);
    state.invalidate();
    boolean done = false;
    // E-step
    int iter;

    PrintWriter out = IOUtils.openOutHard(Execution.getFile(label + ".events"));

    double oldObjective = Double.NEGATIVE_INFINITY;

    for (iter = 0; iter < numIters && !done; iter++) {
      state.invalidate();

      // Logging stuff
      List<String> items = new ArrayList<>();
      items.add("iter = " + iter);
      items.add("objective = " + state.value());
      items.add("pointNorm = " + MatrixOps.norm(state.point()));
      items.add("gradientNorm = " + MatrixOps.norm(state.gradient()));
      LogInfo.logs( StrUtils.join(items, "\t") );
      out.println( StrUtils.join(items, "\t") );
      out.flush();

      double objective = state.value();
      assert objective > oldObjective;
      oldObjective = objective;

      done = maximizer.takeStep(state);
    }
    // Do a gradient check only at the very end.
    learning.common.Utils.doGradientCheck(state);

    List<String> items = new ArrayList<>();
    items.add("iter = " + iter);
    items.add("objective = " + state.value());
    items.add("pointNorm = " + MatrixOps.norm(state.point()));
    items.add("gradientNorm = " + MatrixOps.norm(state.gradient()));
    LogInfo.logs( StrUtils.join(items, "\t") );
    out.println( StrUtils.join(items, "\t") );
    out.flush();

    LogInfo.end_track();
    return done && iter == 1;  // Didn't make any updates
  }

  /**
   * min_{l,X} \|T - l_i X_i \otimes X_i \otimes X_i\|_F^2 + \|P - l_i X_i \otimes X_i \|_F^2 + \|X\|^2_F.
   * @param T
   * @param K
   * @return
   */
  @Override
  public Pair<SimpleMatrix, SimpleMatrix> symmetricOrthogonalFactorize(FullTensor T, int K) {
    double[] lambda = new double[K];
    Arrays.fill(lambda, 1.0/K);
    SymmetricStateVectors state = new SymmetricStateVectors(T, K, lambda);

    LBFGSMaximizer maximizer = new LBFGSMaximizer(backtrack, lbfgs);
    optimize(maximizer, state, "JointTensor", 100);

    throw new RuntimeException("Not yet implemented");
  }

  @Override
  public Pair<SimpleMatrix, SimpleMatrix> symmetricFactorize(FullTensor T, int K) {
    // Project down into the moments
    throw new RuntimeException("Not yet implemented");
  }

  @Override
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> asymmetricFactorize(FullTensor T, int K) {
    throw new RuntimeException("Can't handle asymmetric tensors");
  }
}
