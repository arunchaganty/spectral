/**
 * learning.exceptions
 * Arun Chaganty (chaganty@stanford.edu)
 *
 */

package learning.exceptions;

/**
 * An exception when an algorithm fails to recovery paramters due to
 * some technical failure
 */
public class RecoveryFailure extends Exception {
  public RecoveryFailure() {
    super();
  }
  public RecoveryFailure(String message) {
    super(message);
  }
}

