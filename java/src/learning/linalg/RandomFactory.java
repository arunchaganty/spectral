package learning.linalg;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
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
  public static long seed = 0;
  public static Random rand = new Random(seed);

  public static void setSeed(long seed) {
    RandomFactory.seed = seed;
    rand.setSeed( seed );
  }

  /**
   * Generate a random matrix with standard normal entries.
   * @param D
   * @return
   */
  public static SimpleMatrix randn(Random rand, int N, int D) {
    SimpleMatrix X = MatrixFactory.zeros(N,D);
    for( int i = 0; i < N; i++)
      for( int j = 0; j < D; j++)
        X.set( i, j, rand.nextGaussian() );

    return X;
  }
  public static SimpleMatrix randn(int N, int D) {
    return randn( rand, N, D );
  }
  public static double[][] randn_(Random rand, int m, int n) {
    double[][] X = new double[m][n];
    for( int i = 0; i < m; i++)
      for( int j = 0; j < n; j++)
        X[i][j] = rand.nextGaussian();
    return X;
  }

  public static SimpleMatrix rand(int m, int n) {
    return rand( rand, m, n );
  }
  public static SimpleMatrix rand(Random rand, int m, int n) {
    SimpleMatrix X = MatrixFactory.zeros(m,n);
    for( int i = 0; i < m; i++)
      for( int j = 0; j < n; j++)
        X.set( i, j, 2.0 * rand.nextDouble() - 1.0 );
    return X;
  }
  public static double[][] rand_(Random rand, int m, int n) {
    double[][] X = new double[m][n];
    for( int i = 0; i < m; i++)
      for( int j = 0; j < n; j++)
        X[i][j] = 2.0 * rand.nextDouble() - 1.0;
    return X;
  }
  public static double[] rand_(Random rand, int m) {
    return rand_(rand, 1, m)[0];
  }

  /**
   * Generate a single random variable
   * @param sigma - noise
   * @return
   */
  public static double randn(double sigma2) {
    return rand.nextGaussian() * Math.sqrt(sigma2);
  }

  /**
   * Generate a single random integer variable
   * @param lowerBound - lower bound
   * @param upperBound - upper bound
   * @return
   */
  public static int randInt(int lowerBound, int upperBound) {
    assert( lowerBound < upperBound );
    return lowerBound + rand.nextInt(upperBound - lowerBound);
  }
  public static int randInt(int upperBound) {
    return randInt(0, upperBound);
  }

  public static double randUniform(double lowerBound, double upperBound) {
    return lowerBound + rand.nextDouble() * (upperBound - lowerBound);
  }
  public static double randUniform(double upperBound) {
    return randUniform(0, upperBound);
  }
  public static double randUniform() {
    return randUniform(0, 1);
  }

  public static double randUniform(Random rand, double lowerBound, double upperBound) {
    return lowerBound + rand.nextDouble() * (upperBound - lowerBound);
  }
  public static double randUniform(Random rand, double upperBound) {
    return randUniform(rand, 0, upperBound);
  }
  public static double randUniform(Random rand) {
    return randUniform(rand, 0, 1);
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

  /**
   * Generate a D x D x D tensor of rank K (w.h.p.)
   * @param K
   * @param D
   * @return
   */
  public static FullTensor symmetricTensor(int K, int D) {
    SimpleMatrix w = RandomFactory.rand(1, K);
    // Normalize
    w = MatrixOps.normalize(w.elementMult(w));

    SimpleMatrix X = RandomFactory.rand(D, K);
    return FullTensor.fromDecomposition(w, X);
  }
  public static FullTensor symmetricTensor(int D) {
    return symmetricTensor(D, D);
  }

  public static FullTensor uniformTensor(int D1, int D2, int D3) {
    // Normalize
    FullTensor X = new FullTensor(D1, D2, D3);
    for( int d1 = 0; d1 < D1; d1++ )
      for( int d2 = 0; d2 < D2; d2++ )
        for( int d3 = 0; d3 < D3; d3++ )
          X.X[d1][d2][d3] = randUniform(-1, 1);
    return X;
  }

  public static void symmetric(int N, DenseMatrix64F X) {
    for( int i = 0; i < N; i++ ) {
      for( int j = 0; j < i; j++ ) {
        X.set(i, j, rand.nextGaussian() );
        X.set(j, i, rand.nextGaussian() );
      }
      X.set(i, i, rand.nextGaussian() );
    }
  }
  public static SimpleMatrix symmetric(int N) {
    DenseMatrix64F X = new DenseMatrix64F(N,N);
    symmetric(N, X);
    return SimpleMatrix.wrap(X);
  }

  /**
   * Returns the permutation on n integers
   * @param n - size of permutation
   * @return list of size n to permute
   */
  public static List<Integer> permutation(int n, Random rnd) {
    List<Integer> perm = new ArrayList<Integer>();
    for( int i = 0; i < n; i++ ) perm.add(i);
    Collections.shuffle(perm, rnd);
    return perm;
  }


}

