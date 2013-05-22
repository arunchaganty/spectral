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

  public TensorMethod() {}
  public TensorMethod(int iters, int attempts) {
    this.iters = iters;
    this.attempts = attempts;
  }

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

  /**
   * Reduce the 3-view mixture model to 1 symmetric view.
   */
  public static Pair<SimpleMatrix,FullTensor> symmetrizeViews( int K, 
        SimpleMatrix M12, 
        SimpleMatrix M13, 
        SimpleMatrix M23, 
        FullTensor M123 ) {

    int D = M12.numRows();

    Triplet<SimpleMatrix,SimpleMatrix,SimpleMatrix> U1WU2 = MatrixOps.svdk( M12, K );
    Triplet<SimpleMatrix,SimpleMatrix,SimpleMatrix> U2WU3 = MatrixOps.svdk( M23, K );
    SimpleMatrix U1 = U1WU2.getValue0(); // d x k
    SimpleMatrix U2 = U1WU2.getValue2();
    SimpleMatrix U3 = U2WU3.getValue2();

    assert( U1.numRows() == K );
    assert( U1.numCols() == D );

    // \tilde M_{12} = U_1^T M_{12} U_2
    SimpleMatrix M12_ = // k x k
      U1.transpose().mult // k x d
      (M12).mult(U2); // d x k
    SimpleMatrix M12_i = M12_.invert();

    // P = M_{31} U_1^T (\tilde M_{21})^{-1} U_2 M_{23}
    SimpleMatrix Pairs = // d x d 
        M13.transpose().mult // d x d 
        (U1).mult // d x k 
         (M12_i.transpose()).mult // k x k
         (U2.transpose()).mult // k x d
         (M23);  // d x d 

    // T = M_{123}( M_{32} U_2^T (\tilde M_{12})^{-1} U_1, M_{31} U_1^T (\tilde M_{21})^{-1} U_2,  I )
    FullTensor Triples = 
      M123.rotate(
        (M23.transpose().mult // d x d
          (U2).mult // d x k
          (M12_i).mult // k x k
          (U1.transpose())).transpose(), // k x d
        (M13.transpose().mult
          (U1).mult
          (M12_i.transpose()).mult // k x k
          (U2.transpose())).transpose(), // k x d
        MatrixFactory.eye(D)
        );

    return new Pair<>(Pairs, Triples);
  } 

}
