package learning.models;

import fig.basic.BipartiteMatcher;
import fig.basic.Indexer;
import fig.basic.ListUtils;
import fig.basic.Pair;
import learning.models.loglinear.BinaryFeature;
import learning.models.loglinear.Feature;
import learning.models.loglinear.UnaryFeature;

import java.io.Serializable;
import java.util.*;

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

  public abstract Indexer<Feature> getFeatureIndexer();
  /**
   * To double array
   */
  public abstract double[] toArray();

  /**
   * Replace with other
   */
  public Params merge(Params other_) {
    Indexer<Feature> jointIndexer = new Indexer<>();
    for(Feature feature : getFeatureIndexer().getObjects()) {
      jointIndexer.getIndex(feature);
    }
    for(Feature feature : other_.getFeatureIndexer().getObjects()) {
      jointIndexer.getIndex(feature);
    }

    double[] weights_ = other_.toArray();

    // Now merge
    Params joint = new BasicParams(numGroups(), jointIndexer);
    //noinspection MismatchedReadAndWriteOfArray
    double [] weights = joint.toArray();
    for(Feature feature : getFeatureIndexer().getObjects()) {
      weights[jointIndexer.indexOf(feature)] = toArray()[jointIndexer.indexOf(feature)];
    }
    for(Feature feature : other_.getFeatureIndexer().getObjects()) {
      weights[jointIndexer.indexOf(feature)] = weights_[jointIndexer.indexOf(feature)];
    }

    return joint;
  }


  public void initRandom(Random random, double noise) {
    double[] weights = toArray();
    for (int j = 0; j < weights.length; j++)
      weights[j] = noise * (2 * random.nextDouble() - 1);
  }

  public double get(Feature featureName) {
    return toArray()[getFeatureIndexer().indexOf(featureName)];
  }
  public void set(Feature featureName, double value) {
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
    Indexer<Feature> indexer = getFeatureIndexer();
    Indexer<Feature> indexer_ = other.getFeatureIndexer();
    double[] weights = toArray();
    double[] weights_ = other.toArray();
    if(indexer == indexer_)
      System.arraycopy(weights_, 0, weights, 0, weights.length);
    else { // Copy over slowly.
      for(int i = 0; i < weights.length; i++) {
        Feature feature = indexer.getObject(i);
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
    Indexer<Feature> indexer = getFeatureIndexer();
    Indexer<Feature> indexer_ = other.getFeatureIndexer();
    double[] weights = toArray();
    double[] weights_ = other.toArray();
    if(indexer == indexer_)
      for(int i = 0; i < weights.length; i++)
        weights[i] += scale * weights_[i];
    else { // Copy over slowly.
      for(int i = 0; i < weights.length; i++) {
        Feature feature = indexer.getObject(i);
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
    Indexer<Feature> indexer = getFeatureIndexer();
    Indexer<Feature> indexer_ = other.getFeatureIndexer();
    double[] weights = toArray();
    double[] weights_ = other.toArray();

    if(indexer == indexer_)
      for(int i = 0; i < weights.length; i++)
        prod +=  weights[i] * weights_[i];
    else { // Copy over slowly.
      for(int i = 0; i < weights.length; i++) {
        Feature feature = indexer.getObject(i);
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
    Indexer<Feature> indexer = getFeatureIndexer();
    double[] weights = toArray();
    for (int f = 0; f < size(); f++)
      builder.append(indexer.getObject(f)).append("\t").append(weights[f]).append(" ");
    return builder.toString();
  }



  /**
   * Compute differences in Params with optimal permutation of
   * parameters, ignoring error on unmeasured measured features (of
   * this).
   */
  public double computeDiff(Params that, int[] perm) {
    // Compute differences in ParamsVec with optimal permutation of parameters.
    // Assume features have the form h=3,..., where the label '3' can be interchanged with another digit.
    // Use bipartite matching.
    int K = numGroups();

    Indexer<Feature> indexer = this.getFeatureIndexer();
    Indexer<Feature> indexer_ = that.getFeatureIndexer();
//    assert(indexer == indexer_);

    double[][] costs = new double[K][K];  // Cost if assign latent state h1 of this to state h2 of that
    for (int j = 0; j < size(); j++) {
      Feature rawFeature = indexer.getObject(j);
      if (!(rawFeature instanceof UnaryFeature)) continue;
      UnaryFeature feature = (UnaryFeature)rawFeature;

      int h1 = feature.h;
      double v1 = get(feature);
      for (int h2 = 0; h2 < K; h2++) {
        try {
        double v2 = that.get(new UnaryFeature(h2, feature.description));
        costs[h1][h2] += Math.abs(v1-v2);
        } catch(ArrayIndexOutOfBoundsException ignored) {}
      }
    }

    if(perm == null) perm = new int[K];
    // Find the permutation that minimizes cost.
    BipartiteMatcher matcher = new BipartiteMatcher();
    ListUtils.set(perm, matcher.findMinWeightAssignment(costs));

    // Compute the actual cost (L1 error).
    double cost = 0;
    for (int j = 0; j < size(); j++) {
      Feature rawFeature = indexer.getObject(j);
      if (rawFeature instanceof BinaryFeature) {
        BinaryFeature feature = (BinaryFeature)rawFeature;
        try {
        double v1 = this.get(feature);
        double v2 = that.get(new BinaryFeature(perm[feature.h1], perm[feature.h2]));
        cost += Math.abs(v1 - v2);
        } catch(ArrayIndexOutOfBoundsException ignored) {}
      } else {
        UnaryFeature feature = (UnaryFeature)rawFeature;
        try {
        double v1 = this.get(feature);
        double v2 = that.get(new UnaryFeature(perm[feature.h], feature.description));
        cost += Math.abs(v1-v2);
        } catch(ArrayIndexOutOfBoundsException ignored) {}
      }
    }
    return cost;
  }

  public double computeDiff2(Params that, int[] perm) {
    // Compute differences in ParamsVec with optimal permutation of parameters.
    // Assume features have the form h=3,..., where the label '3' can be interchanged with another digit.
    // Use bipartite matching.
    int K = numGroups();

    Indexer<Feature> indexer = this.getFeatureIndexer();
    Indexer<Feature> indexer_ = that.getFeatureIndexer();
//    assert(indexer == indexer_);

    double[][] costs = new double[K][K];  // Cost if assign latent state h1 of this to state h2 of that
    for (int j = 0; j < size(); j++) {
      Feature rawFeature = indexer.getObject(j);
      if (!(rawFeature instanceof UnaryFeature)) continue;
      UnaryFeature feature = (UnaryFeature)rawFeature;

      int h1 = feature.h;
      double v1 = get(feature);
      for (int h2 = 0; h2 < K; h2++) {
        try {
          double v2 = that.get(new UnaryFeature(h2, feature.description));
          costs[h1][h2] += (v1-v2) * (v1-v2);
        } catch(ArrayIndexOutOfBoundsException ignored) {}
      }
    }

    if(perm == null) perm = new int[K];
    // Find the permutation that minimizes cost.
    BipartiteMatcher matcher = new BipartiteMatcher();
    ListUtils.set(perm, matcher.findMinWeightAssignment(costs));

    // Compute the actual cost (L1 error).
    double cost = 0;
    for (int j = 0; j < size(); j++) {
      Feature rawFeature = indexer.getObject(j);
      if (rawFeature instanceof BinaryFeature) {
        BinaryFeature feature = (BinaryFeature)rawFeature;
        try {
          double v1 = this.get(feature);
          double v2 = that.get(new BinaryFeature(perm[feature.h1], perm[feature.h2]));
          cost += (v1 - v2) * (v1 - v2);
        } catch(ArrayIndexOutOfBoundsException ignored) {}
      } else {
        UnaryFeature feature = (UnaryFeature)rawFeature;
        try {
          double v1 = this.get(feature);
          double v2 = that.get(new UnaryFeature(perm[feature.h], feature.description));
          cost += (v1-v2) * (v1-v2);
        } catch(ArrayIndexOutOfBoundsException ignored) {}
      }
    }
    return Math.sqrt(cost);
  }

  /**
   * @return the number of groups
   */
  public int numGroups() {
    throw new RuntimeException("Not implemented yet");
  }


}
