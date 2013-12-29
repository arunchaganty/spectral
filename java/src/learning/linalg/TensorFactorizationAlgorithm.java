package learning.linalg;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

/**
 * Created by chaganty on 12/16/13.
 */
public interface TensorFactorizationAlgorithm {

  /**
   * Decompose the symmetric tensor t into K orthogonal components
   * @param T
   * @param K
   * @return (\lambda: eigenvalues, V: eigenvectors): $T ~= \sum \lambda_i V_i^{\otimes3}$
   */
  public Pair<SimpleMatrix, SimpleMatrix> symmetricFactorize(FullTensor T, int K);

  /**
   * Decompose an arbitrary tensor T into K orthogonal components
   * @param T
   * @param K
   * @return (\lambda: eigenvalues, V: eigenvectors): $T ~= \sum \lambda_i V_i^{\otimes3}$
   */
  public Pair<SimpleMatrix, SimpleMatrix> factorize(FullTensor T, int K);
}
