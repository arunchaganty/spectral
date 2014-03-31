package learning.linalg;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.javatuples.Quartet;

/**
 * Interface for tensor factorization algorithms
 */
public interface TensorFactorizationAlgorithm {

  /**
   * Decompose the symmetric tensor t into K orthogonal components
   * @param T
   * @param K
   * @return (\lambda: eigenvalues, V: eigenvectors): $T ~= \sum \lambda_i V_i^{\otimes3}$
   */
  public Pair<SimpleMatrix, SimpleMatrix> symmetricOrthogonalFactorize(FullTensor T, int K);

  /**
   * Decompose an arbitrary tensor T into K orthogonal components
   * @param T
   * @param K
   * @return (\lambda: eigenvalues, V: eigenvectors): $T ~= \sum \lambda_i V_i^{\otimes3}$
   */
  public Pair<SimpleMatrix, SimpleMatrix> symmetricFactorize(FullTensor T, int K);


  /**
   * Decompose an asymmetric tensor t into K orthogonal components
   * @param T - tensor
   * @param K - number of components
   * @return (\lambda: eigenvalues, V_1, V_2, V_3: eigenvectors): $T ~= \sum \lambda_i V_i^{\otimes3}$
   */
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> asymmetricFactorize(FullTensor T, int K);

}
