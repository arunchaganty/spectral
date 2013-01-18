package learning.linalg;

import java.util.Random;

import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.factory.QRDecomposition;
import org.ejml.simple.SimpleMatrix;

import fig.prob.MultGaussian;

/**
 * A set of functions to generate random variables
 */
public class RandomFactory {
  public static Random rand = new Random();
  public int seed = 0;

  public static void setSeed(long seed) {
    rand.setSeed( seed );
  }

  /**
   * Generate a random matrix with standard normal entries.
   * @param D
   * @return
   */
  public static SimpleMatrix randn(int N, int D) {
    SimpleMatrix X = MatrixFactory.zeros(N,D);
    for( int i = 0; i < N; i++)
      for( int j = 0; j < D; j++)
        X.set( i, j, rand.nextGaussian() );

    return X;
  }

  public static SimpleMatrix rand(int m, int n) {
    SimpleMatrix X = MatrixFactory.zeros(m,n);
    for( int i = 0; i < m; i++)
      for( int j = 0; j < n; j++)
        X.set( i, j, 2.0 * rand.nextDouble() - 1.0 );

    return X;
  }

  /**
   * Generate a single random variable
   * @param sigma - noise
   * @return
   */
  public static double randn(double sigma) {
    return rand.nextGaussian() * sigma;
  }

  /**
   * Generate a random orthogonal 'd' dimensional matrix, using the
   * the technique described in: Francesco Mezzadri, "How to generate 
   * random matrices from the classical compact groups" 
   * @param d
   * @return
   */
  public static SimpleMatrix orthogonal(int d) {
    SimpleMatrix Z = randn(d,d);
    QRDecomposition<DenseMatrix64F> Z_QR = DecompositionFactory.qr(Z.numRows(), Z.numCols());
    Z_QR.decompose(Z.getMatrix());
    SimpleMatrix Q = SimpleMatrix.wrap( Z_QR.getQ(null, true) );
    SimpleMatrix R = SimpleMatrix.wrap( Z_QR.getR(null, true) ); 
    SimpleMatrix D = MatrixFactory.diag(R);
    for( int i = 0; i < d; i++)
      D.set(i, D.get(i)/Math.abs(D.get(i)));
    return Q.mult(MatrixFactory.diag(D));
  }

  /**
   * Draw an element from a multinomial distribution with weights given in matrix.
   * @param pi
   * @return
   */
  public static int multinomial(SimpleMatrix pi) {
    return multinomial( pi.getMatrix().data );
  }

  /**
   * Draw an element from a multinomial distribution with weights given in matrix.
   * @param pi
   * @return
   */
  public static int multinomial(double[] pi) {
    double x = rand.nextDouble();
    for( int i = 0; i < pi.length; i++ )
    {
      if( x <= pi[i] )
        return i;
      else
        x -= pi[i];
    }

    // The remaining probability is assigned to the last element in the sequence.
    return pi.length-1;
  }

  /**
   * Draw many elements from a multinomial distribution with weights given in matrix.
   * @param n - Number of draws
   * @param pi - Parameters
   * @return - Vector with count of number of times a value was drawn
   */
  public static double[] multinomial(double[] pi, int n) {
    double[] cnt = new double[pi.length];

    for( int i = 0; i < n; i++)
      cnt[ multinomial(pi) ] += 1;

    return cnt;
  }
  public static SimpleMatrix multinomial(SimpleMatrix pi, int n) {
    return MatrixFactory.fromVector( multinomial( MatrixFactory.toVector( pi ), n ) );
  }

  /**
   * Generate a random matrix with standard normal entries.
   * @param D
   * @return
   */
  public static double[][] multivariateGaussian(double[] mean, double[][] cov, int count) {
    MultGaussian gaussian = new MultGaussian( mean, cov );

    double[][] X = new double[count][mean.length];
    for( int i = 0; i < count; i++) {
      double[] y = gaussian.sample( rand );
      for( int j = 0; j < mean.length; j++ )
        X[i][j] = y[j];
    }

    return X;
  }
  public static SimpleMatrix multivariateGaussian(SimpleMatrix mean, SimpleMatrix cov, int count) {
    double[] mean_ = MatrixFactory.toVector( mean );
    double[][] cov_ = MatrixFactory.toArray( cov );

    return new SimpleMatrix( multivariateGaussian( mean_, cov_, count ) );
  }

}

