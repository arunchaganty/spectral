/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.linalg;
import org.ejml.simple.SimpleMatrix;

public interface Tensor {
  /**
   * Project the tensor onto a matrix by taking an inner product with theta
   * @param axis
   * @param theta
   * @return
   */
  public SimpleMatrix project( int axis, SimpleMatrix theta );
  public SimpleMatrix project2( int axis1, int axis2, SimpleMatrix theta1, SimpleMatrix theta2 );
  public double project3(SimpleMatrix theta1, SimpleMatrix theta2, SimpleMatrix theta3);
  public int getDim( int axis );

}

