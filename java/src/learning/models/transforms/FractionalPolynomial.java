package learning.models.transforms;

import learning.linalg.RandomFactory;

import java.util.Random;

import static learning.Misc.*;

/**
 * Basis generated using random real powers.
 */
public class FractionalPolynomial extends NonLinearity {
  public int degree;
  public double upperBound;

  /**
   * Create a fractional polynomial with degree terms.
   * @param degree - the number of terms in the output term
   * @param upperBound - an upper bound on the monomial exponents.
   */
  public FractionalPolynomial(int degree, double upperBound) {
    this.degree = degree;
    this.upperBound = upperBound;
  }
  public FractionalPolynomial(int degree) {
    this(degree, 10.0);
  }

  /**
   * Return the number of dimensions in the linearized version of the
   * non-linearity.
   */
  @Override
  public int getLinearDimension(int dimension) {
    return degree * dimension;
  }

  @Override
  public double[] getLinearEmbedding(double[] x) {
    final int D = x.length;
    final int D_ = getLinearDimension(D);

    final double[] y = new double[D_];
    // The first D rows are x itself.
    for( int d = 0; d < D; d++ ) {
      y[d] = x[d];
    }

    // The remaining D rows are random pairwise and increasing fractional powers
    for( int d = D; d < D_; d++ ) {
      int k1 = RandomFactory.randInt(0, D);
      double p1 = RandomFactory.randUniform(1.2, upperBound);
      y[d] = Math.signum(x[k1]) * Math.pow(Math.abs(x[k1]), p1);
      if( D > 1 ) {
        int k2 = (k1 + RandomFactory.randInt(1, D)) % D;
        double p2 = RandomFactory.randUniform(1.2, upperBound);
        y[d] *= Math.signum(x[k2]) * Math.pow(Math.abs(x[k2]), p2);
      }
    }

    return y;
  }


}
