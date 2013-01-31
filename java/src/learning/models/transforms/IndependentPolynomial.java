package learning.models.transforms;

import fig.basic.LogInfo;
import learning.Misc;
import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;

import java.util.ArrayList;
import java.util.Collection;

/**
 * Generate a set of quadratically and cubically independent polynomials using monomial terms
 */
public class IndependentPolynomial extends NonLinearity {
  double degree;
  int dimension;
  int[][] exponents;

  public IndependentPolynomial(int degree, int dimension) {
    this.degree = degree;
    this.dimension = dimension;
    this.exponents = computeExponents(dimension, degree);
    MatrixOps.printArray(exponents);
  }

  public static boolean isQuadraticallyIndependent(Collection<int[]> set, int[] candidate) {
    int D = candidate.length;
    for( int[] x1 : set ) {
      for( int[] x2 : set ) {
        assert( x1.length ==  D);
        // Check if this exponent is the sum of any other two or vice versa.
        boolean isSum = true;
        boolean isSummed1 = true;
        boolean isSummed2 = true;
        for(int d = 0; d < D; d++) {
          isSum = isSum & (x1[d] + x2[d] == candidate[d]);
          isSummed1 = isSummed1 & (x2[d] + candidate[d] == x1[d]);
          isSummed2 = isSummed2 & (x1[d] + candidate[d] == x2[d]);
        }
        if( isSum || isSummed1 || isSummed2 ) {
          return false;
        }
      }
    }
    return true;
  }
  public static boolean isCubicallyIndependent(Collection<int[]> set, int[] candidate) {
    int D = candidate.length;
    for( int[] x1 : set ) {
      for( int[] x2 : set ) {
        for( int[] x3 : set ) {
          assert( x1.length ==  D);
          // Check if this exponent is the sum of any other two or vice versa.
          boolean isSum = true, isSummed1 = true, isSummed2 = true, isSummed3 = true;
          for(int d = 0; d < D; d++) {
            isSum = isSum & (x1[d] + x2[d] + x3[d] == candidate[d]);
            isSummed1 = isSummed1 & (x2[d] + x3[d] + candidate[d] == x1[d]);
            isSummed2 = isSummed2 & (x1[d] + x3[d] + candidate[d] == x2[d]);
            isSummed3 = isSummed3 & (x1[d] + x2[d] + candidate[d] == x3[d]);
          }
          if( isSum || isSummed1|| isSummed2 || isSummed3 ) {
            return false;
          }
        }
      }
    }
    return true;
  }

  /**
   * Compute a set of quadratically and cubically independent polynomials
   * with at most 'degree' terms.
   * @param degree
   * @return
   */
  public static int[][] computeExponents(int dimension, int degree) {
    // Compute the possible exponents using a sieve
    // NOTE: By extension of J. Steinhardt's probabilistic method arguments,
    // it should be (deg)^4, but that sounds pretty bad, so let's lop off
    // a degree and terminate early
    final int D = dimension; int P = degree;
    final double selectionProb = 1.0; ///Math.pow(P,1.0);
    final ArrayList<int[]> exponents = new ArrayList<>();

    // Add a bias
    // exponents.add( new int[D] );
    // Add the first D elements
//    for(int d = 0; d < D; d++ ) {
//      int[] values = new int[D];
//      values[d] = 1;
//      exponents.add( values );
//    }

    // Try each of the P choices of degree D
    Misc.traverseChoices(D, P, new Misc.TraversalFunction() {
      @Override
      public void run(int[] values) {
        if( MatrixOps.sum(values) < 1 ) return;
        // With probability, choose this exponent
        if(RandomFactory.randUniform() < selectionProb) {
          boolean qi = isQuadraticallyIndependent(exponents, values);
          boolean ci = isCubicallyIndependent(exponents, values);
          // Great! This is linearly independent!
          if( qi && ci ) {
            // IMPORTANT: Make a copy!
            exponents.add(values.clone());
          }
        }
      }
    });

    return exponents.toArray(new int[][] {});
  }

  /**
   * Return the number of dimensions in the linearized version of the
   * non-linearity.
   */
  @Override
  public int getLinearDimension(int dimension) {
    assert( dimension == this.dimension );
    return exponents.length;
  }

  @Override
  public double[] getLinearEmbedding(final double[] x) {
    int D = x.length;
    int D_ = getLinearDimension(D);

    double[] y = new double[D_];

    for(int d_ = 0; d_ < D_; d_++ ) {
      y[d_] = 1.0;
      for(int d = 0; d < D; d++ ) {
        y[d_] *= Math.pow(x[d], exponents[d_][d]);
      }
    }
    return y;
  }
}
