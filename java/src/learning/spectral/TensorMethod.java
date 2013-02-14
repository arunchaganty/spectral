package learning.spectral;

import fig.basic.Option;
import learning.linalg.*;

import org.javatuples.*;

import org.ejml.simple.SimpleMatrix;
import fig.basic.LogInfo;

/**
 * Recover parameters from symmetric multi-views using the Tensor powerup method
 */
public class TensorMethod {

  @Option
  int iters = 1000;
  @Option
  int attempts = 10;

  /**
   * The tensor factorization method is just finding
   * the eigenvalues/eigenvectors of the tensor Triples.
   * @param K
   * @param Pairs
   * @param Triples
   * @return
   */
  public Pair<SimpleMatrix,SimpleMatrix> recoverParameters( int K, SimpleMatrix Pairs, FullTensor Triples ) {
    Pair<SimpleMatrix, SimpleMatrix> pair = TensorFactorization.eigendecompose(Triples, Pairs, K, attempts, iters);
    return pair;
  }
}
