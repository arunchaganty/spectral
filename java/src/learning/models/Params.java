package learning.models;

import fig.basic.Indexer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

import static learning.common.Utils.writeStringHard;

/**
 * Uniform interface for (vectorizable)-parameters
 */
public abstract class Params implements Serializable {

  /**
   * Create another params with the same configuration
   * @return new empty params
   */
  public abstract Params newParams();

  public abstract Indexer<String> getFeatureIndexer();
  /**
   * To double array
   */
  public abstract double[] toArray();

  /**
   * Replace with other
   */
  public Params merge(Params other_) {
    Indexer<String> jointIndexer = new Indexer<>();
    for(String feature : getFeatureIndexer().getObjects()) {
      jointIndexer.getIndex(feature);
    }
    for(String feature : other_.getFeatureIndexer().getObjects()) {
      jointIndexer.getIndex(feature);
    }

    double[] weights_ = other_.toArray();

    // Now merge
    Params joint = new BasicParams(jointIndexer);
    //noinspection MismatchedReadAndWriteOfArray
    double [] weights = joint.toArray();
    for(String feature : getFeatureIndexer().getObjects()) {
      weights[jointIndexer.indexOf(feature)] = toArray()[jointIndexer.indexOf(feature)];
    }
    for(String feature : other_.getFeatureIndexer().getObjects()) {
      weights[jointIndexer.indexOf(feature)] = weights_[jointIndexer.indexOf(feature)];
    }

    return joint;
  }


  public void initRandom(Random random, double noise) {
    double[] weights = toArray();
    for (int j = 0; j < weights.length; j++)
      weights[j] = noise * (2 * random.nextDouble() - 1);
  }

  public double get(String featureName) {
    return toArray()[getFeatureIndexer().indexOf(featureName)];
  }
  public void set(String featureName, double value) {
    toArray()[getFeatureIndexer().indexOf(featureName)] = value;
  }

  public int size() {
    return toArray().length;
  }
  public void clear() {
    Arrays.fill(toArray(), 0.);
  }

  /**
   * Replace with other
   */
  public void copyOver(Params other) {
    // The weights are compatible - copy!
    Indexer<String> indexer = getFeatureIndexer();
    Indexer<String> indexer_ = other.getFeatureIndexer();
    double[] weights = toArray();
    double[] weights_ = other.toArray();
    if(indexer == indexer_)
      System.arraycopy(weights_, 0, weights, 0, weights.length);
    else { // Copy over slowly.
      for(int i = 0; i < weights.length; i++) {
        String feature = indexer.getObject(i);
        if(indexer_.contains(feature))
          weights[i] = weights_[indexer_.indexOf(feature)];
      }
    }
  }

  // Algebraic operations
  /**
   * Update by adding other with scale
   */
  public void plusEquals(double scale, Params other) {
    // The weights are compatible - copy!
    Indexer<String> indexer = getFeatureIndexer();
    Indexer<String> indexer_ = other.getFeatureIndexer();
    double[] weights = toArray();
    double[] weights_ = other.toArray();
    if(indexer == indexer_)
      for(int i = 0; i < weights.length; i++)
        weights[i] += scale * weights_[i];
    else { // Copy over slowly.
      for(int i = 0; i < weights.length; i++) {
        String feature = indexer.getObject(i);
        if(indexer_.contains(feature))
          weights[i] += scale * weights_[indexer_.indexOf(feature)];
      }
    }
  }
  /**
   * Update by scaling each entry
   */
  public void scaleEquals(double scale) {
    double[] weights = toArray();
    for(int i = 0; i < weights.length; i++)
      weights[i] *= scale;
  }

  /**
   * Take the dot product of two params
   */
  public double dot(Params other) {
    double prod = 0.;

    // The weights are compatible - copy!
    Indexer<String> indexer = getFeatureIndexer();
    Indexer<String> indexer_ = other.getFeatureIndexer();
    double[] weights = toArray();
    double[] weights_ = other.toArray();

    if(indexer == indexer_)
      for(int i = 0; i < weights.length; i++)
        prod +=  weights[i] * weights_[i];
    else { // Copy over slowly.
      for(int i = 0; i < weights.length; i++) {
        String feature = indexer.getObject(i);
        if(indexer_.contains(feature))
          prod += weights[i] * weights_[indexer_.indexOf(feature)];
      }
    }

    return prod;
  }

  // TODO: Support matching

  /**
   * Create another params with the same configuration
   * @return new params with same entries as other
   */
  public Params copy() {
    Params other = newParams();
    other.copyOver(this);
    return other;
  }
  public void plusEquals(Params other) {
    plusEquals(1.0, other);
  }
  /**
   * Update by creating a new object and adding
   */
  public Params plus(double scale, Params other) {
    Params ret = newParams();
    ret.plusEquals(scale, other);
    return ret;
  }
  public Params plus(Params other) {
    return plus(1.0, other);
  }
  /**
   * Update by scaling each entry
   */
  public Params scale(double scale) {
    Params ret = newParams();
    ret.scaleEquals(scale);
    return ret;
  }

  public Params restrict(Params other) {
    Params ret = newParams();
    ret.copyOver(other);
    return ret;
  }

  public void write(String path) {
    writeStringHard(path, toString());
  }

  public void cache() {
  }
  public void invalidateCache() {
  }
  public boolean isCacheValid() {
    return false;
  }

  public String toString() {
    StringBuilder builder = new StringBuilder();
    Indexer<String> indexer = getFeatureIndexer();
    double[] weights = toArray();
    for (int f = 0; f < size(); f++)
      builder.append(indexer.getObject(f)).append("\t").append(weights[f]).append(" ");
    return builder.toString();
  }
}
