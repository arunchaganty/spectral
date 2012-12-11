/**
 * learning.exceptions
 * Arun Chaganty (chaganty@stanford.edu)
 *
 */

package learning.exceptions;

/**
 * An algorithm or procedure fails due to numerical instability or
 * an otherwise ill-conditioned state.
 */
public class NumericalException extends Exception {
  public NumericalException() {
    super();
  }
  public NumericalException(String message) {
    super(message);
  }
}

