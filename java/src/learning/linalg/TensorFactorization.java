package learning.linalg;

import fig.basic.LogInfo;
import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;
import org.javatuples.Pair;

/**
 * Implements Anandkumar/Ge/Hsu/Kakade/Telgarsky, 2012.
 * Method of moments for learning latent-variable models.
 */
public class TensorFactorization {
  static final double EPS_CLOSE = 1e-10;

  /**
   * Perform a single eigen-decomposition step
   * @param T
   * @return
   */
  protected static Pair<Double,SimpleMatrix> eigendecomposeStep( Tensor T, int attempts, int iters ) {
    int N = iters;
    int D = T.getDim(0);

    double maxEigenvalue = Double.NEGATIVE_INFINITY;
    SimpleMatrix maxEigenvector = null;

    for( int attempt = 0; attempt < attempts; attempt++ ) {
      SimpleMatrix theta = RandomFactory.randn(1, D);
      theta.scale(1.0/MatrixOps.norm(theta));

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
   * @param scale
   * @param v
   * @return
   */
  protected static void deflate(FullTensor T, double scale, double[] v) {
    int D = T.getDim(0);
    for( int d1 = 0; d1 < D; d1++ )
      for( int d2 = 0; d2 < D; d2++ )
        for( int d3 = 0; d3 < D; d3++ )
          T.set(d1,d2,d3, T.get(d1, d2, d3) - scale * v[d1] * v[d2] * v[d3] );
  }
  protected static void deflate(FullTensor T, double scale, SimpleMatrix vector) {
    deflate(T, scale, MatrixFactory.toVector(vector) );
  }

  /**
   * Find the first $k$ eigen-vectors and eigen-values of the tensor $T = \sum \lambda_i v_i^{\otimes 3}$
   * @param T
   * @param K - largest K eigenvalues will be returned
   * @return eigenvalues and eigenvectors
   */
  public static Pair<SimpleMatrix, SimpleMatrix> eigendecompose( FullTensor T, int K, int attempts, int iters ) {
    int D = T.getDim(0);

    SimpleMatrix[] eigenvectors = new SimpleMatrix[K];
    double[] eigenvalues = new double[K];

    // Make a copy of T because we're going to destroy it during deflation
    T = T.clone();

    for( int k = 0; k < K; k++ ) {
      // Extract the top eigenvalue/vector pair
      Pair<Double, SimpleMatrix> pair = eigendecomposeStep(T, attempts, iters);

      // When null, then we are done
      if( pair.getValue1() == null )
        break;

      eigenvalues[k] = pair.getValue0();
      eigenvectors[k] = pair.getValue1();

      // Deflate
      deflate(T, eigenvalues[k], eigenvectors[k]);
    }

    return new Pair<>(MatrixFactory.fromVector(eigenvalues), MatrixFactory.columnStack(eigenvectors));
  }
  public static Pair<SimpleMatrix, SimpleMatrix> eigendecompose( FullTensor T, int K ) {
    return eigendecompose(T, K, 10, 100);
  }
  public static Pair<SimpleMatrix, SimpleMatrix> eigendecompose( FullTensor T ) {
    int K = T.getDim(0);
    return eigendecompose(T, K);
  }

  /**
   * Whiten and decompose
   * @param T - Tensor
   * @param P - Second-moments
   * @return
   */
  public static Pair<SimpleMatrix, SimpleMatrix> eigendecompose( FullTensor T, SimpleMatrix P, int K, int attempts, int iters ) {
    LogInfo.begin_track("tensor-eigendecomposition");

    // Whiten
    SimpleMatrix W = MatrixOps.whitener(P, K);
    LogInfo.logs( "W: " + W );
    SimpleMatrix Winv = MatrixOps.colorer(P, K);
    FullTensor Tw = T.rotate(W,W,W);

    Pair<SimpleMatrix, SimpleMatrix> pair = eigendecompose(Tw, K, attempts, iters);

    // Color them in again
    SimpleMatrix eigenvalues = pair.getValue0();
    SimpleMatrix eigenvectors = pair.getValue1();

    assert( K == eigenvalues.getNumElements() );
    int D = Tw.getDim(0);

    // TODO: Remove
    // Make sure this was a factorization
    FullTensor Tw_ = FullTensor.fromDecomposition( eigenvalues, eigenvectors );
    LogInfo.logs( "Tw_: " + MatrixOps.diff(Tw, Tw_) );


    // Scale the vectors by 1/sqrt(eigenvalues);
    {
      for( int k = 0; k < K; k++ )
        for( int d = 0; d < D; d++ )
          eigenvectors.set(d,k, eigenvectors.get(d,k) * eigenvalues.get(k) ) ;
    }
    eigenvectors = Winv.mult( eigenvectors );

    // Eigenvalues are w^{-1/2}; w is what we want.
    for(int i = 0; i < K; i++)
      eigenvalues.set( i, Math.pow(eigenvalues.get(i), -2) );

    FullTensor T_ = FullTensor.fromDecomposition( eigenvalues, eigenvectors );
    LogInfo.logs( "T_: " + MatrixOps.diff(T, T_) );

    LogInfo.end_track("tensor-eigendecomposition");
    return new Pair<>(eigenvalues, eigenvectors);
  }
  public static Pair<SimpleMatrix, SimpleMatrix> eigendecompose( FullTensor T, SimpleMatrix P, int K ) {
    return eigendecompose(T, P, K, 10, 100 );
  }
  public static Pair<SimpleMatrix, SimpleMatrix> eigendecompose( FullTensor T, SimpleMatrix P ) {
    int K = MatrixOps.rank(P);
    return eigendecompose(T, P, K);
  }

  /**
   * Check to see that v is indeed an eigenvector of T
   * @param T
   * @param v
   * @return
   */
  protected static boolean isEigenvector( Tensor T, SimpleMatrix v ) {
    // Firstly, normalize $v$ - to be sure.
    v = MatrixOps.normalize(v);
    SimpleMatrix u = T.project2( 1, 2, v, v );
    u = MatrixOps.normalize(u);
    return MatrixOps.allclose( u, v );
  }

}
