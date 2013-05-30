/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

import java.util.List;

public interface Featurizer {
  /**
   * Returns a sparse featurized version of the data
   */
  public List<Integer> features(String word);
  public int numFeatures();
}

