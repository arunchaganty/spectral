package learning.models.transforms;

import learning.linalg.RandomFactory;

import java.util.Random;

import static learning.Misc.*;

/**
 * Basis generated using random real powers.
 */
public class FractionalPolynomial extends NonLinearity {
  public int degree;
  public int dimension;
  public double[][] exponents;

  /**
   * Create a fractional polynomial with degree terms.
   * @param degree - the number of terms in the output term
   * @param upperBound - an upper bound on the monomial exponents.
   */
  public FractionalPolynomial(int degree, int dimension, double upperBound) {
    this.degree = degree;
    this.dimension = dimension;
    // Generate exponents
    this.exponents = computeExponents(degree, dimension, 1.2, upperBound);
  }
  public FractionalPolynomial(int degree, int dimension ) {
    this(degree, dimension, 10.0);
  }

  public static double[][] computeExponents( int degree, int dimension, double lowerBound, double upperBound ) {
    int D = dimension;
    int D_ = degree * dimension;
    double[][] exponents = new double[ D_ ][D];
    for(int d = 0; d < D; d++) {
      double[] exp = new double[D]; exp[d] = 1;
      exponents[d] = exp;
    }
    for( int d = D; d < D_; d++ ) {
      double[] exp = new double[D];
      int d1 = RandomFactory.randInt(0, D);
      exp[d1] = RandomFactory.randUniform(1.2, upperBound);
      if( D > 1 ) {
        int d2 = (d1 + RandomFactory.randInt(1, D)) % D;
        exp[d2] = RandomFactory.randUniform(1.2, upperBound);
      }
      exponents[d] = exp;
    }

    return exponents;
  }

  @Override
  public double[][] getExponents() {
    return exponents;
  }

  /**
   * Return the number of dimensions in the linearized version of the
   * non-linearity.
   */
  @Override
  public int getLinearDimension(int dimension) {
    assert( dimension == this.dimension );
    return degree * dimension;
  }

  @Override
  public double[] getLinearEmbedding(double[] x) {
    final int D = x.length;
    final int D_ = getLinearDimension(D);

    final double[] y = new double[D_];
    // The first D rows are x itself.
    for( int d_ = 0; d_ < D_; d_++ ) {
      y[d_] = 1.0;
      for( int d = 0; d < D; d++ ) {
        y[d_] *= Math.signum(x[d]) * Math.pow(Math.abs(x[d]), exponents[d_][d]);
      }
    }

    return y;
  }


}
