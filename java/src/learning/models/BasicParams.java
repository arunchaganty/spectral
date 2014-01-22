package learning.models;

import fig.basic.Indexer;

import java.util.Random;

/**
 * Basic set of parameters
 */
public class BasicParams extends Params {

  Indexer<String> featureIndexer;
  double[] weights;

  public BasicParams(Indexer<String> featureIndexer) {
    this.featureIndexer = featureIndexer;
    weights = new double[featureIndexer.size()];
  }

  @Override
  public Params newParams() {
    return new BasicParams(featureIndexer);
  }

  @Override
  public Indexer<String> getFeatureIndexer() {
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
}
