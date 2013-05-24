package learning.spectral;

import fig.basic.Option;
import learning.linalg.*;
import learning.data.MomentAggregator;

import org.javatuples.*;
import java.util.*;

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
   * The tensor factorization method is just finding
   * the eigenvalues/eigenvectors of the tensor Triples.
   * @param K
   * @param Pairs
   * @param Triples
   * @return
   */
  public Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> 
      recoverParameters( int K, SimpleMatrix M12, 
          SimpleMatrix M13, SimpleMatrix M23, 
          FullTensor M123 ) {
    LogInfo.begin_track("recovery-asymmetric");
    // Symmetrize views to get M33, M333
    Pair<SimpleMatrix,FullTensor> symmetricMoments = symmetrizeViews( K, M12, M13, M23, M123 );
    SimpleMatrix Pairs = symmetricMoments.getValue0();
    FullTensor Triples = symmetricMoments.getValue1();

    // Tensor Factorize to get w, M3
    Pair<SimpleMatrix, SimpleMatrix> pair = recoverParameters( K, Pairs, Triples );
    SimpleMatrix pi = pair.getValue0();
    SimpleMatrix M3 = pair.getValue1();

    // Invert M3 to get M1 and M2.

    SimpleMatrix inversion = (M3.transpose()).pseudoInverse().mult( MatrixFactory.diag( MatrixOps.reciprocal(pi) ) );

    SimpleMatrix M1 = M13.mult( inversion );
    SimpleMatrix M2 = M23.mult( inversion );
    LogInfo.end_track("recovery-asymmetric");

    return new Quartet<>( pi, M1, M2, M3 );
  }
  public Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> 
      recoverParameters( int K, 
        Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,FullTensor> moments ) {
        return recoverParameters( K,
            moments.getValue0(),
            moments.getValue1(),
            moments.getValue2(),
            moments.getValue3() );
    }


  /**
   * Reduce the 3-view mixture model to 1 symmetric view.
   */
  public static Pair<SimpleMatrix,FullTensor> symmetrizeViews( int K, 
        SimpleMatrix M12, 
        SimpleMatrix M13, 
        SimpleMatrix M23, 
        FullTensor M123 ) {
    LogInfo.begin_track("symmetrize-views");

    int D = M12.numRows();

    Triplet<SimpleMatrix,SimpleMatrix,SimpleMatrix> U1WU2 = MatrixOps.svdk( M12, K );
    Triplet<SimpleMatrix,SimpleMatrix,SimpleMatrix> U2WU3 = MatrixOps.svdk( M23, K );
    SimpleMatrix U1 = U1WU2.getValue0(); // d x k
    SimpleMatrix U2 = U1WU2.getValue2();
    SimpleMatrix U3 = U2WU3.getValue2();

    MatrixOps.printSize( U1 );

    assert( U1.numRows() == D );
    assert( U1.numCols() == K );

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

    LogInfo.end_track("symmetrize-views");
    return new Pair<>(Pairs, Triples);
  } 


  /**
   * Extract parameters right from a data sequence.
   */
  public Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> 
      recoverParameters( int K, int D, Iterator<double[][]> dataSeq ) {

      // Compute moments 
      MomentAggregator agg = new MomentAggregator(D, dataSeq);
      agg.run();

      return recoverParameters( K, agg.getMoments() );
    }
}
