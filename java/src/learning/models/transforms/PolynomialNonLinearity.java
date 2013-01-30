package learning.models.transforms;

import learning.Misc;

import static learning.Misc.binomial;
import static learning.Misc.traverseMultiCombination;

/**
 * Non-linear functions represented using a polynomial basis
 */
public class PolynomialNonLinearity extends NonLinearity {
  public int degree;

  public PolynomialNonLinearity(int degree) {
    this.degree = degree;
  }

  public PolynomialNonLinearity() {
    this(1);
  }

  /**
   * Return the number of dimensions in the linearized version of the
   * non-linearity.
   */
  @Override
  public int getLinearDimension(int dimension) {
    return binomial(dimension + degree - 1, degree);
  }

  /**
   * Return the linear embedding of $x$, like x_1^2 + x_1 x_2 + x_2
   * x_1 + x_2^2.
   * The output is in lexicographic ordering with x_1 occupying the
   * first index.
   */
  @Override
  public double[] getLinearEmbedding(final double[] x) {
    final int D = x.length;
    final int D_ = getLinearDimension(D);
    final double[] y = new double[D_];

    traverseMultiCombination(D, degree,
            new Misc.TraversalFunction() {
              int d_ = 0;

              @Override
              public void run(int[] powers) {
                y[d_] = 1.0;
                for (int d = 0; d < D; d++) {
                  y[d_] *= Math.pow(x[d], powers[d]);
                }
                d_++;
              }
            });
    return y;
  }
}
