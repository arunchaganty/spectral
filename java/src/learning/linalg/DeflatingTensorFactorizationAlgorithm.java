package learning.linalg;

import fig.basic.*;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.javatuples.Quartet;

import java.util.Random;

/**
 * Abstracts deflating methods
 */
public abstract class DeflatingTensorFactorizationAlgorithm implements TensorFactorizationAlgorithm {

  protected abstract Pair<Double,SimpleMatrix> eigendecomposeStep( FullTensor T );

  /**
   * Return T - scale vector^{\otimes 3}
   * "Efficient" inplace deflation
   * @param scale - remove $v$ with this scale
   * @param vector - vector to remove
   */
  protected void deflate(FullTensor T, double scale, SimpleMatrix vector) {
    int D = T.getDim(0);
    for( int d1 = 0; d1 < D; d1++ )
      for( int d2 = 0; d2 < D; d2++ )
        for( int d3 = 0; d3 < D; d3++ )
          T.set(d1,d2,d3, T.get(d1, d2, d3) - scale * vector.get(d1) * vector.get(d2) * vector.get(d3) );
  }

  /**
   * Find the first $k$ eigen-vectors and eigen-values of the tensor $T = \sum \lambda_i v_i^{\otimes 3}$
   * @param T - Tensor to be decomposed
   * @param K - largest K eigenvalues will be returned
   * @return eigenvalues and eigenvectors
   */
  @Override
  public Pair<SimpleMatrix, SimpleMatrix> symmetricOrthogonalFactorize(FullTensor T, int K) {
    SimpleMatrix[] eigenvectors = new SimpleMatrix[K];
    double[] eigenvalues = new double[K];

    // Make a copy of T because we're going to destroy it during deflation
    FullTensor T_ = T.clone();

    for( int k = 0; k < K; k++ ) {
      // Extract the top eigenvalue/vector pair
      LogInfo.logs("Eigenvector %d/%d", k, K);
      Pair<Double, SimpleMatrix> pair = eigendecomposeStep(T_);

      // When null, then we are done
      if( pair.getValue1() == null )
        break;

      eigenvalues[k] = pair.getValue0();
      eigenvectors[k] = pair.getValue1();

      // Deflate
      deflate(T_, eigenvalues[k], eigenvectors[k]);
    }

    SimpleMatrix eigenvalues_ = MatrixFactory.fromVector(eigenvalues);
    SimpleMatrix eigenvectors_ = MatrixFactory.columnStack(eigenvectors);

    // Make sure we have a factorization.
    {
      assert( K == eigenvalues_.getNumElements() );
      FullTensor Treconstructed = FullTensor.fromDecomposition( eigenvalues_, eigenvectors_ );
      LogInfo.logs( "T_: " + MatrixOps.diff(T, Treconstructed) );
    }

    return Pair.with(eigenvalues_, eigenvectors_);
  }

  /**
   * Check to see that v is indeed an eigenvector of T
   */
  protected boolean isEigenvector( Tensor T, SimpleMatrix v ) {
    // Firstly, normalize $v$ - to be sure.
    v = MatrixOps.normalize(v);
    SimpleMatrix u = T.project2( 1, 2, v, v );
    u = MatrixOps.normalize(u);
    return MatrixOps.allclose( u, v );
  }

  @Override
  public Pair<SimpleMatrix, SimpleMatrix> symmetricFactorize(FullTensor T, int K) {
    // TODO: Implement whitening routine here
    throw new RuntimeException("Can't handle unsymmetric tensors");
  }

  @Override
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> asymmetricFactorize(FullTensor T, int K) {
    throw new RuntimeException("Can't handle asymmetric tensors");
  }

}
