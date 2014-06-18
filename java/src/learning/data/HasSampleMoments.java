package learning.data;

import learning.linalg.FullTensor;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

/**
 * Computes the moments using sample estimates.
 */
public interface HasSampleMoments {
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> computeSampleMoments(int N);
}
