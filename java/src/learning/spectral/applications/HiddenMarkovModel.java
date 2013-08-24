/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import learning.spectral.TensorMethod;
import learning.linalg.*;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.*;

/**
 * Word clustering using a HMM model
 *    + Reads data as a sequence of real-valued vectors, or a moment matrix
 *      of the same
 *    + Runs tensor-factorization and does post processing to find
 *    emission and transition probabilities.
 */
public class HiddenMarkovModel implements Runnable {

  /**
   * Runs tensor factorization on the moments of the data and returns
   * (\pi, T, O).
   * Note that unless the moments are calculated using only the first 3
   * words, $\pi$ and $T$ are not exact.
   */
  public Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix>
      run( int K, 
          SimpleMatrix M12, SimpleMatrix M13, SimpleMatrix M23, FullTensor M123 ) {
    // Tensor Factorize
    TensorMethod algo = new TensorMethod();
    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> params = algo.recoverParameters( K, M12, M13, M23, M123 );

    // T \pi
    SimpleMatrix Tpi = params.getValue0();
    // O
    SimpleMatrix O = params.getValue2();
    // O T
    SimpleMatrix OT = params.getValue3();

    SimpleMatrix T = O.pseudoInverse().mult(OT);
    SimpleMatrix pi = T.invert().mult(Tpi);

    // Return emission and transition probabilities.

    return new Triplet<>( pi, T, O );
  }

  public void run() {}

}

