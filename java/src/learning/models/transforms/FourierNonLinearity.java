package learning.models.transforms;

import learning.Misc;
import static learning.Misc.*;

/**
 * Fourier basis set for non linearities
 */
public class FourierNonLinearity extends NonLinearity {
  public int degree;

  public FourierNonLinearity(int degree) {
    this.degree = degree;
  }

  public FourierNonLinearity() {
    this(1);
  }

  /**
   * Return the number of dimensions in the linearized version of the
   * non-linearity.
   */
  @Override
  public int getLinearDimension(int dimension) {
    return (int) Math.pow(degree, dimension);
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

    traverseChoices(D, degree,
            new Misc.TraversalFunction() {
              int d_ = 0;

              @Override
              public void run(int[] periods) {
                y[d_] = 1.0;

                double z = 0;
                for (int d = 0; d < D; d++) {
                  z += x[d] * periods[d];
                }
                y[d_++] = Math.cos(z);
              }
            });
    return y;
  }
}
