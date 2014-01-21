package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import learning.models.Params;

public class ParamsVec extends Params {
  public final int K;  // Number of hidden states
  public final Indexer<Feature> featureIndexer;
  public final Indexer<String> stringFeatureIndexer;
  public final int numFeatures;
  public final double[] weights;

  static Indexer<String> constructStringFeatureIndexer(Indexer<Feature> other) {
    Indexer<String> indexer = new Indexer<>();
    for(Feature f : other.getObjects()) // getObjects is guaranteed to have a consistent order.
      indexer.add(f.toString());
    return indexer;
  }

  ParamsVec(int K, Indexer<Feature> featureIndexer, Indexer<String> stringFeatureIndexer) {
    this.K = K;
    this.featureIndexer = featureIndexer;
    this.stringFeatureIndexer = stringFeatureIndexer;
    this.numFeatures = featureIndexer.size();
    this.weights = new double[numFeatures];
  }

  // -- Params implementation
  @Override
  public Params newParams() {
    return new ParamsVec(K, featureIndexer, stringFeatureIndexer);
  }
  @Override
  public Params merge(Params that_) {
    if(that_ instanceof  ParamsVec) {
      ParamsVec that = (ParamsVec) that_;
      // Join the two featureIndexers
      Indexer<Feature> joinedIndex = new Indexer<>();
      joinedIndex.addAll(featureIndexer);
      joinedIndex.addAll(that.featureIndexer);
      ParamsVec ret = new ParamsVec(K, joinedIndex, constructStringFeatureIndexer(joinedIndex));
      for( Feature f : featureIndexer )
        ret.weights[ret.featureIndexer.indexOf(f)] += weights[featureIndexer.indexOf(f)];
      for( Feature f : that.featureIndexer )
        ret.weights[ret.featureIndexer.indexOf(f)] += that.weights[that.featureIndexer.indexOf(f)];
      return ret;
    } else {
      throw new IllegalArgumentException();
    }
  }
  @Override
  public double[] toArray() {return weights;}
  @Override
  public int size() {return weights.length;}
  @Override
  public void clear() { Arrays.fill(weights,0.); }

  public Indexer<String> getFeatureIndexer() {
    return stringFeatureIndexer;
  }

  // -- ParamsVec specific
  public double get(Feature f) {
    return this.weights[featureIndexer.indexOf(f)];
  }
  public void set(Feature f, double val) {
    this.weights[featureIndexer.indexOf(f)] = val;
  }
  public void incr(Feature f, double val) {
    this.weights[featureIndexer.indexOf(f)] += val;
  }

//  /**
//   * Add two params vecs and place result in vec3
//   * @param vec1
//   * @param vec2
//   * @param vec3
//   * @return
//   */
//  public static ParamsVec plus(ParamsVec vec1, ParamsVec vec2, ParamsVec vec3 ) {
//    for( Feature f : vec3.featureIndexer ) {
//      vec3.weights[vec3.featureIndexer.indexOf(f)] =
//              (vec1.featureIndexer.contains(f) ? vec1.weights[vec1.featureIndexer.indexOf(f)] : 0.)
//                      + (vec2.featureIndexer.contains(f) ? vec2.weights[vec2.featureIndexer.indexOf(f)] : 0.);
//    }
//
//    return vec3;
//  }
//  public static ParamsVec plus(ParamsVec vec1, ParamsVec vec2 ) {
//    Indexer<Feature> featureIndexer = new Indexer<>();
//    featureIndexer.addAll(vec1.featureIndexer);
//    featureIndexer.addAll(vec2.featureIndexer);
//
//    ParamsVec vec3 = new ParamsVec(vec1.K, featureIndexer);
//    plus( vec1, vec2, vec3 );
//    return vec3;
//  }

//  /**
//   * Subtract two params vecs and place result in vec3
//   * @param vec1
//   * @param vec2
//   * @param vec3
//   * @return
//   */
//  public static ParamsVec minus(ParamsVec vec1, ParamsVec vec2, ParamsVec vec3 ) {
//    for( Feature f : vec3.featureIndexer ) {
//      vec3.weights[vec3.featureIndexer.indexOf(f)] =
//              (vec1.featureIndexer.contains(f) ? vec1.weights[vec1.featureIndexer.indexOf(f)] : 0.)
//                      - (vec2.featureIndexer.contains(f) ? vec2.weights[vec2.featureIndexer.indexOf(f)] : 0.);
//    }
//
//    return vec3;
//  }
//  public static ParamsVec minus(ParamsVec vec1, ParamsVec vec2 ) {
//    Indexer<Feature> featureIndexer = new Indexer<>();
//    featureIndexer.addAll(vec1.featureIndexer);
//    featureIndexer.addAll(vec2.featureIndexer);
//
//    ParamsVec vec3 = new ParamsVec(vec1.K, featureIndexer);
//    minus( vec1, vec2, vec3 );
//    return vec3;
//  }
  public double computeDiff(ParamsVec that, int[] perm) {
    // Compute differences in ParamsVec with optimal permutation of parameters.
    // Assume features have the form h=3,..., where the label '3' can be interchanged with another digit.
    // Use bipartite matching.

    double[][] costs = new double[K][K];  // Cost if assign latent state h1 of this to state h2 of that
    for (int j = 0; j < numFeatures; j++) {
      Feature rawFeature = featureIndexer.getObject(j);
      if (!(rawFeature instanceof UnaryFeature)) continue;
      UnaryFeature feature = (UnaryFeature)rawFeature;
      int h1 = feature.h;
      double v1 = this.weights[j];
      for (int h2 = 0; h2 < K; h2++) {
        double v2 = that.weights[featureIndexer.indexOf(new UnaryFeature(h2, feature.description))];
        costs[h1][h2] += Math.abs(v1-v2);
      }
    }

    // Find the permutation that minimizes cost.
    BipartiteMatcher matcher = new BipartiteMatcher();
    ListUtils.set(perm, matcher.findMinWeightAssignment(costs));

    // Compute the actual cost (L1 error).
    double cost = 0;
    for (int j = 0; j < numFeatures; j++) {
      Feature rawFeature = featureIndexer.getObject(j);
      if (rawFeature instanceof BinaryFeature) {
        BinaryFeature feature = (BinaryFeature)rawFeature;
        int perm_j = featureIndexer.indexOf(new BinaryFeature(perm[feature.h1], perm[feature.h2]));
        cost += Math.abs(this.weights[j] - that.weights[perm_j]);
        continue;
      }
      UnaryFeature feature = (UnaryFeature)rawFeature;
      int h1 = feature.h;
      double v1 = this.weights[j];
      int h2 = perm[h1];
      double v2 = that.weights[featureIndexer.indexOf(new UnaryFeature(h2, feature.description))];
      cost += Math.abs(v1-v2);
    }
    return cost;
  }

  /**
   * Compute differences in ParamsVec with optimal permutation of
   * parameters, ignoring error on unmeasured measured features (of
   * this).
   */
  double computeDiff(ParamsVec that, boolean[] measuredFeatures, int[] perm) {
    // Assume features have the form h=3,..., where the label '3' can be interchanged with another digit.
    // Use bipartite matching.

    double[][] costs = new double[K][K];  // Cost if assign latent state h1 of this to state h2 of that
    for (int j = 0; j < numFeatures; j++) {
      if(!measuredFeatures[j]) continue;

      Feature rawFeature = featureIndexer.getObject(j);
      if (!(rawFeature instanceof UnaryFeature)) continue;
      UnaryFeature feature = (UnaryFeature)rawFeature;
      int h1 = feature.h;
      double v1 = this.weights[j];
      // Only initialize if it is a measured feature.
      for (int h2 = 0; h2 < K; h2++) {
        double v2 = that.weights[featureIndexer.indexOf(new UnaryFeature(h2, feature.description))];
        costs[h1][h2] += Math.abs(v1-v2);
      }
    }

    if (perm == null) perm = new int[K];
    // Find the permutation that minimizes cost.
    BipartiteMatcher matcher = new BipartiteMatcher();
    ListUtils.set(perm, matcher.findMinWeightAssignment(costs));

    // Compute the actual cost (L1 error).
    double cost = 0;
    for (int j = 0; j < numFeatures; j++) {
      if(!measuredFeatures[j]) continue;

      Feature rawFeature = featureIndexer.getObject(j);
      if (rawFeature instanceof BinaryFeature) {
        BinaryFeature feature = (BinaryFeature)rawFeature;
        int perm_j = featureIndexer.indexOf(new BinaryFeature(perm[feature.h1], perm[feature.h2]));
        cost += Math.abs(this.weights[j] - that.weights[perm_j]);
      } else {
        UnaryFeature feature = (UnaryFeature)rawFeature;
        int h1 = feature.h;
        double v1 = this.weights[j];
        int h2 = perm[h1];
        double v2 = that.weights[featureIndexer.indexOf(new UnaryFeature(h2, feature.description))];
        cost += Math.abs(v1-v2);
      }
    }
    return cost;
  }

  public void write(String path) {
    PrintWriter out = IOUtils.openOutHard(path);
    //for (int f : ListUtils.sortedIndices(weights, true))
    for (int f = 0; f < numFeatures; f++)
      out.println(featureIndexer.getObject(f) + "\t" + weights[f]);
    out.close();
  }

  public String toString() {
    StringBuilder builder = new StringBuilder();
    for (int f = 0; f < numFeatures; f++)
      builder.append(featureIndexer.getObject(f)).append("\t").append(weights[f]).append(" ");
    return builder.toString();
  }
}
