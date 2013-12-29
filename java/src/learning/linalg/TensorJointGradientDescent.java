package learning.linalg;

import fig.basic.BacktrackingLineSearch;
import fig.basic.LBFGSMaximizer;
import fig.basic.Maximizer;
import fig.basic.OptionSet;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

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


  /**
   * min_{l,X} \|T - l_i X_i \otimes X_i \otimes X_i\|_F^2 + \|P - l_i X_i \otimes X_i \|_F^2 + \|X\|^2_F.
   * @param T
   * @param K
   * @return
   */
  @Override
  public Pair<SimpleMatrix, SimpleMatrix> symmetricFactorize(FullTensor T, int K) {

  }

  @Override
  public Pair<SimpleMatrix, SimpleMatrix> factorize(FullTensor T, int K) {
    // Project down into the moments
    throw new RuntimeException("Not yet implemented");
  }
}
