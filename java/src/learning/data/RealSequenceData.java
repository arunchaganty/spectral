/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

public interface RealSequenceData {
  /**
   * Return the dimensionality of the sequence data.
   */
  public int getDimension();
  public int getInstanceCount();
  public int getInstanceLength( int instance );
  public double[] getDatum( int instance, int index );
}

