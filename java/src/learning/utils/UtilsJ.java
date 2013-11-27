package learning.utils;

import fig.basic.Maximizer;
import learning.data.ComputableMoments;
import learning.data.HasExactMoments;
import learning.linalg.FullTensor;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

/**
 * Various utilities
 */
public class UtilsJ {
  public static ComputableMoments fromExactMoments(HasExactMoments obj) {
    final Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments = obj.computeExactMoments();
    return new ComputableMoments() {
      @Override
      public MatrixOps.Matrixable computeP13() {
        return MatrixOps.matrixable( moments.getValue0() );
      }

      @Override
      public MatrixOps.Matrixable computeP12() {
        return MatrixOps.matrixable( moments.getValue1() );
      }

      @Override
      public MatrixOps.Matrixable computeP32() {
        return MatrixOps.matrixable( moments.getValue2() );
      }

      @Override
      public MatrixOps.Tensorable computeP123() {
        return MatrixOps.tensorable( moments.getValue3() );
      }
    };
  }

  public static void doGradientCheck(Maximizer.FunctionState state) {
    double epsilon = 1e-4;
    // Save point
    double[] point = state.point();
    double[] gradient = state.gradient();
    double[] currentGradient = gradient.clone();
    double[] currentPoint = point.clone();


    // Set point to be +/- gradient
    for( int i = 0; i < currentPoint.length; i++ ) {
      point[i] = currentPoint[i] + epsilon;
      double valuePlus = state.value();
      point[i] = currentPoint[i] - epsilon;
      double valueMinus = state.value();
      point[i] = currentPoint[i];

      double expectedValue = (valuePlus - valueMinus)/(2*epsilon);
      double actualValue = currentGradient[i];
      assert MatrixOps.equal(expectedValue, actualValue, 1e-4);
    }
  }

}
