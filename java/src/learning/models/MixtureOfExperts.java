/**
 * learning.model
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.RandomFactory;

import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;

import java.util.Random;

/**
 * A mixture of experts model
 */
public class MixtureOfExperts {
  protected int K; // Number of experts
  protected int D; // Dimensionality of space

  // Model parameters
  protected SimpleMatrix weights;
  protected SimpleMatrix betas;
  protected double sigma2;

  // Point generator
  protected SimpleMatrix mean;
  protected SimpleMatrix cov;

  Random rnd = new Random();

  public MixtureOfExperts( int K, int D, SimpleMatrix weights, SimpleMatrix betas, double sigma2, SimpleMatrix mean, SimpleMatrix cov ) {
    this.K = K;
    this.D = D;

    this.weights = weights;
    this.betas = betas;
    this.sigma2 = sigma2;

    this.mean = mean;
    this.cov = cov;
  }

  /**
   * Sample n points from the point distribution and generate y's
   * according to the mixture of experts model
   *
   * @return The returned value is a tuple of 2 elements, the first
   * containing $y$ and the second $X$.
   */
  public SimpleMatrix[] sample( int N ) {
    // Generate n random points
    SimpleMatrix X = RandomFactory.multivariateGaussian( mean, cov, N );
    double[][] X_ = MatrixFactory.toArray( X );

    // Get the betas in a row form to make it easier to generate data
    double[][] betas_ = MatrixFactory.toArray( betas.transpose() );

    double[] y = new double[ N ];
    for( int n = 0; n < N; n++ ) {
      // Sample the coordinates $z$
      int z = (int) RandomFactory.multinomial( weights );

      // Generate the y's
      y[n] = MatrixOps.dot( betas_[z], X_[n] );
      if( sigma2 > 0 )
        y[n] += RandomFactory.randn( sigma2 );
    }

    SimpleMatrix[] result = {MatrixFactory.fromVector( y ), X};

    return result;
  }

  public static enum WeightDistribution {
    Uniform,
      Random
  }

  public static enum BetaDistribution {
    Eye,
      Random
  }

  public static enum MeanDistribution {
    Zero,
      Random
  }

  public static enum CovarianceDistribution {
    Eye,
      Spherical,
      Random
  }

  /**
   * Generate a MixtureOfExperts model
   * @param K - number of experts
   * @param D - number of dimensions
   * @param sigma2 - noise in y
   * @param wDistribution - distribution on the mixture weights
   * @param bDistribution - distribution on the betas
   * @param mDistribution - distribution on the point means
   * @param SDistribution - distribution on the point variance
   * @param pointSigma - variance parameter for the points.
   */
  public static MixtureOfExperts generate( final int K, final int D, double sigma2, WeightDistribution wDistribution, BetaDistribution bDistribution, MeanDistribution mDistribution, CovarianceDistribution SDistribution, double pointSigma ) {
    double[] w = new double[K];
    double[][] B = new double[D][K];
    double[] M = new double[D];
    double[][] S = new double[D][D];

    switch( wDistribution ) {
      case Uniform:
        for(int i = 0; i < K; i++ ) w[i] = 1.0/K;
        break;
      case Random:
        // Generate random values, and then normalise
        for(int i = 0; i < K; i++ ) w[i] = Math.abs( RandomFactory.randn(1.0) ); 
        MatrixOps.normalize( w );
        break;
    }

    switch( bDistribution ) {
      case Eye:
        for(int i = 0; i < D; i++) {
          for(int j = 0; j < D; j++) {
            B[i][j] = (i == j) ? 1.0 : 0.0;
          }
        }
        break;
      case Random:
        for(int i = 0; i < D; i++) {
          for(int j = 0; j < D; j++) {
            B[i][j] = RandomFactory.randn(1.0);
          }
        }

    }

    switch( mDistribution ) {
      case Zero:
        break;
      case Random:
        for( int i = 0; i < D; i++ )
          M[i] = RandomFactory.randn(1.0);
    }

    switch( SDistribution ) {
      case Eye:
        for(int i = 0; i < D; i++) {
          for(int j = 0; j < D; j++) {
            S[i][j] = (i == j) ? pointSigma : 0.0;
          }
        }
        break;
      case Spherical:
        for(int i = 0; i < D; i++) {
          for(int j = 0; j < D; j++) {
            S[i][j] = (i == j) ? Math.abs( RandomFactory.randn(pointSigma) ) : 0.0;
          }
        }
        break;
      case Random:
        throw new NoSuchMethodError();
    }

    SimpleMatrix weights = MatrixFactory.fromVector( w );
    SimpleMatrix betas = new SimpleMatrix( B );
    SimpleMatrix mean = MatrixFactory.fromVector( M );
    SimpleMatrix cov = new SimpleMatrix( B );

    return new MixtureOfExperts( K, D, weights, betas, sigma2, mean, cov );
  }

  public static MixtureOfExperts generate( final int K, final int D, double sigma2, WeightDistribution wDistribution, BetaDistribution bDistribution, MeanDistribution mDistribution, CovarianceDistribution SDistribution ) {
    return generate( K, D, sigma2, wDistribution, bDistribution, mDistribution, SDistribution, 10.0 );
  }


}

