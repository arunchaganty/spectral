/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

public class RealSequence implements RealSequenceData {
  protected final double[][][] data;

  public RealSequence( double[][][] data ) {
    this.data = data;
  }

  public int getDimension() {
    return data[0][0].length;
  }
  public int getInstanceCount() {
    return data.length;
  }
  public int getInstanceLength( int instance ) {
    return data[instance].length;
  }
  public double[] getDatum( int instance, int index ) {
    return data[instance][index];
  }
}

