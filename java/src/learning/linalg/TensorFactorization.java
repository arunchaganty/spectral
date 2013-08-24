package learning.linalg;

import fig.basic.LogInfo;
import fig.basic.Option;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import java.util.Random;

/**
 * Implements Anandkumar/Ge/Hsu/Kakade/Telgarsky, 2012.
 * Method of moments for learning latent-variable models.
 */
public class TensorFactorization {
  static final double EPS_CLOSE = 1e-10;

  @Option(gloss="Random number generator for tensor method and random projections")
  Random rnd = new Random();

  /**
   * Perform a single eigen-decomposition step
   * @param T - Full tensor
   * @return - (eigenvalue, eigenvector).
   */
  protected Pair<Double,SimpleMatrix> eigendecomposeStep( Tensor T, int attempts, int iters ) {
    int N = iters;
    int D = T.getDim(0);

    double maxEigenvalue = Double.NEGATIVE_INFINITY;
    SimpleMatrix maxEigenvector = null;

    for( int attempt = 0; attempt < attempts; attempt++ ) {
      SimpleMatrix theta = RandomFactory.randn(rnd, 1, D);
      theta.scale(1.0/MatrixOps.norm(theta));
      if( attempt % 10 == 0 )
        LogInfo.logs("Attempt %d/%d", attempt, attempts);

      // Hit the tensor with this vector and repeat till you converge
      for(int n = 0; n < N; n++ ) {
        SimpleMatrix theta_ = T.project2(1, 2, theta, theta);
        // Normalize
        theta_ = theta_.scale(1.0 / MatrixOps.norm(theta_));
        double err = MatrixOps.norm(theta_.minus(theta));
        if( err < EPS_CLOSE ) {
          theta = theta_; break;
        } else {
          theta = theta_;
        }
      }
      double eigenvalue = T.project3(theta, theta, theta);
      if(eigenvalue > maxEigenvalue) {
        maxEigenvalue = eigenvalue;
        maxEigenvector = theta;
      }
    }

    return new Pair<>(maxEigenvalue, maxEigenvector);
  }

  /**
   * Return T - scale vector^{\otimes 3}
   * "Efficient" inplace deflation
   * @param scale - remove $v$ with this scale
   * @param v - vector to remove
   * @return - Deflated tensor
   */
  protected void deflate(FullTensor T, double scale, double[] v) {
    int D = T.getDim(0);
    for( int d1 = 0; d1 < D; d1++ )
      for( int d2 = 0; d2 < D; d2++ )
        for( int d3 = 0; d3 < D; d3++ )
          T.set(d1,d2,d3, T.get(d1, d2, d3) - scale * v[d1] * v[d2] * v[d3] );
  }
  protected void deflate(FullTensor T, double scale, SimpleMatrix vector) {
    deflate(T, scale, MatrixFactory.toVector(vector) );
  }

  /**
   * Find the first $k$ eigen-vectors and eigen-values of the tensor $T = \sum \lambda_i v_i^{\otimes 3}$
   * @param T - Tensor to be decomposed
   * @param K - largest K eigenvalues will be returned
   * @param attempts - number of times each eigendecomposition step is run (to pick the eigenvector of largest magnitude)
   * @param iters - Number of iterations to run the power method.
   * @return eigenvalues and eigenvectors
   */
  public Pair<SimpleMatrix, SimpleMatrix> eigendecompose( FullTensor T, int K, int attempts, int iters ) {
    int D = T.getDim(0);

    SimpleMatrix[] eigenvectors = new SimpleMatrix[K];
    double[] eigenvalues = new double[K];

    // Make a copy of T because we're going to destroy it during deflation
    T = T.clone();

    for( int k = 0; k < K; k++ ) {
      // Extract the top eigenvalue/vector pair
      LogInfo.logs("Eigenvector %d/%d", k, K);
      Pair<Double, SimpleMatrix> pair = eigendecomposeStep(T, attempts, iters);

      // When null, then we are done
      if( pair.getValue1() == null )
        break;

      eigenvalues[k] = pair.getValue0();
      eigenvectors[k] = pair.getValue1();

      // Deflate
      deflate(T, eigenvalues[k], eigenvectors[k]);
    }

    SimpleMatrix eigenvalues_ = MatrixFactory.fromVector(eigenvalues);
    SimpleMatrix eigenvectors_ = MatrixFactory.columnStack(eigenvectors);;

    {
      assert( K == eigenvalues_.getNumElements() );
      // Make sure this was a factorization
      FullTensor T_ = FullTensor.fromDecomposition( eigenvalues_, eigenvectors_ );
      LogInfo.logs( "T_: " + MatrixOps.diff(T, T_) );
    }

    return Pair.with(eigenvalues_, eigenvectors_);
  }
  public Pair<SimpleMatrix, SimpleMatrix> eigendecompose( FullTensor T, int K ) {
    return eigendecompose(T, K, 10, 100);
  }
  public Pair<SimpleMatrix, SimpleMatrix> eigendecompose( FullTensor T ) {
    int K = T.getDim(0);
    return eigendecompose(T, K);
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

}
