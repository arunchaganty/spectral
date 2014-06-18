package learning.data;

import learning.linalg.FullTensor;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

/**
 * Can compute exact moments
 */
public interface HasExactMoments {
  /**
   * Compute exact moments of the model
   * @return - \E[x_1 x_3^T], \E[x_1 x_2^T], \E[x_3 x_2^T], \E[x_1 \otimes x_2 \otimes x_3]
   */
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> computeExactMoments();
}
