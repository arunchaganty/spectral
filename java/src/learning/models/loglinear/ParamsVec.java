package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

public class ParamsVec {
  public int K;  // Number of hidden states
  public Indexer<Feature> featureIndexer;
  public int numFeatures;
  ParamsVec(int K, Indexer<Feature> featureIndexer) {
    this.K = K;
    this.featureIndexer = featureIndexer;
    this.numFeatures = featureIndexer.size();
    this.weights = new double[numFeatures];
  }

  public double[] weights;

  public ParamsVec(ParamsVec that) {
    this.K = that.K;
    this.featureIndexer = that.featureIndexer;
    this.numFeatures = featureIndexer.size();
    this.weights = that.weights.clone();
  }

  public void copy(ParamsVec that) {
    assert( that.weights.length == this.weights.length );
    System.arraycopy(that.weights, 0, this.weights, 0, this.weights.length);
  }

  public void initRandom(Random random, double noise) {
    for (int j = 0; j < numFeatures; j++)
      weights[j] = noise * (2 * random.nextDouble() - 1);
  }

  public void clear() { ListUtils.set(weights, 0); }
  public void incr(double scale, ParamsVec that) {
    for( Feature f : featureIndexer )
      if( that.featureIndexer.contains(f) )
        this.weights[featureIndexer.getIndex(f)] += scale * that.weights[that.featureIndexer.getIndex(f)];
  }
  public double dot(ParamsVec that) {
    double prod = 0.;
    for( Feature f : featureIndexer )
      if( that.featureIndexer.contains(f) )
        prod += this.weights[featureIndexer.getIndex(f)] * that.weights[that.featureIndexer.getIndex(f)];

    return prod;
  }

  public double get(Feature f) {
    return this.weights[featureIndexer.getIndex(f)];
  }
  public void set(Feature f, double val) {
    this.weights[featureIndexer.getIndex(f)] = val;
  }
  public void incr(Feature f, double val) {
    this.weights[featureIndexer.getIndex(f)] += val;
  }

  /**
   * Add two params vecs and place result in vec3
   * @param vec1
   * @param vec2
   * @param vec3
   * @return
   */
  public static ParamsVec plus(ParamsVec vec1, ParamsVec vec2, ParamsVec vec3 ) {
    for( Feature f : vec3.featureIndexer ) {
      vec3.weights[vec3.featureIndexer.getIndex(f)] =
              (vec1.featureIndexer.contains(f) ? vec1.weights[vec1.featureIndexer.getIndex(f)] : 0.)
                      + (vec2.featureIndexer.contains(f) ? vec2.weights[vec2.featureIndexer.getIndex(f)] : 0.);
    }

    return vec3;
  }
  public static ParamsVec plus(ParamsVec vec1, ParamsVec vec2 ) {
    Indexer<Feature> featureIndexer = new Indexer<>();
    featureIndexer.addAll(vec1.featureIndexer);
    featureIndexer.addAll(vec2.featureIndexer);

    ParamsVec vec3 = new ParamsVec(vec1.K, featureIndexer);
    plus( vec1, vec2, vec3 );
    return vec3;
  }

  /**
   * Subtract two params vecs and place result in vec3
   * @param vec1
   * @param vec2
   * @param vec3
   * @return
   */
  public static ParamsVec minus(ParamsVec vec1, ParamsVec vec2, ParamsVec vec3 ) {
    for( Feature f : vec3.featureIndexer ) {
      vec3.weights[vec3.featureIndexer.getIndex(f)] =
              (vec1.featureIndexer.contains(f) ? vec1.weights[vec1.featureIndexer.getIndex(f)] : 0.)
                      - (vec2.featureIndexer.contains(f) ? vec2.weights[vec2.featureIndexer.getIndex(f)] : 0.);
    }

    return vec3;
  }
  public static ParamsVec minus(ParamsVec vec1, ParamsVec vec2 ) {
    Indexer<Feature> featureIndexer = new Indexer<>();
    featureIndexer.addAll(vec1.featureIndexer);
    featureIndexer.addAll(vec2.featureIndexer);

    ParamsVec vec3 = new ParamsVec(vec1.K, featureIndexer);
    minus( vec1, vec2, vec3 );
    return vec3;
  }

  /**
   * Set all weights of to to be those of from where from has the fields
   * @param from
   * @param to
   * @return
   */
  public static ParamsVec project(ParamsVec from, ParamsVec to ) {
    for( Feature f : to.featureIndexer ) {
      if (from.featureIndexer.contains(f))
        to.weights[to.featureIndexer.getIndex(f)] = from.weights[from.featureIndexer.getIndex(f)];
    }

    return to;
  }

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
      builder.append(featureIndexer.getObject(f) + "\t" + weights[f] + " ");
    return builder.toString();
  }

  public void scale(double scale) {
    for(int i = 0; i < weights.length; i++)
      weights[i] *= scale;
  }
}
