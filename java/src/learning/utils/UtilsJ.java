package learning.utils;

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
}
