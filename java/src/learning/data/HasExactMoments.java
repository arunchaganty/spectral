package learning.data;

import learning.linalg.FullTensor;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

/**
 * Can compute exact moments
 */
public interface HasExactMoments {
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> computeExactMoments();
}
