package learning.models;

import fig.basic.Indexer;
import learning.models.loglinear.Feature;

/**
 * Basic set of parameters
 */
public class BasicParams extends Params {
  final protected Indexer<Feature> featureIndexer;
  public final double[] weights;
  final protected int K;

  public BasicParams(final int K, final Indexer<Feature> featureIndexer) {
    this.K = K;
    this.featureIndexer = featureIndexer;
    weights = new double[featureIndexer.size()];
  }

  @Override
  public Params newParams() {
    return new BasicParams(K, featureIndexer);
  }

  @Override
  public Indexer<Feature> getFeatureIndexer() {
    return featureIndexer;
  }

  @Override
  public double[] toArray() {
    return weights;
  }

  @Override
  public int size() {
    return weights.length;
  }

  @Override
  public int numGroups() {
    return K;
  }

}
