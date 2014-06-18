package learning.data;

import learning.linalg.MatrixOps.Tensorable;
import learning.linalg.MatrixOps.Matrixable;

/**
 * Describes the class as having computable moments.
 */
public interface ComputableMoments {
  public Matrixable computeP13();
  public Matrixable computeP12();
  public Matrixable computeP32();
  public Tensorable computeP123();

}
