package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import learning.models.Params;

public class ParamsVec extends Params {
  public final int K;  // Number of hidden states
  public final Indexer<Feature> featureIndexer;
  public final int numFeatures;
  public final double[] weights;

  ParamsVec(int K, Indexer<Feature> featureIndexer) {
    this.K = K;
    this.featureIndexer = featureIndexer;
    this.numFeatures = featureIndexer.size();
    this.weights = new double[numFeatures];
  }
  public ParamsVec(ParamsVec that) {
    this.K = that.K;
    this.featureIndexer = that.featureIndexer;
    this.numFeatures = featureIndexer.size();
    this.weights = that.weights.clone();
  }

  // -- Params implementation
  @Override
  public Params newParams() {
    return new ParamsVec(K, featureIndexer);
  }
  @Override
  public void copyOver(Params that_) {
    if(that_ instanceof  ParamsVec) {
      ParamsVec that = (ParamsVec) that_;
      if(that.featureIndexer == featureIndexer) {
        // They share the same featureIndexer, so lets optimize
        System.arraycopy(that.weights, 0, this.weights, 0, this.weights.length);
      } else {
        for( Feature f : featureIndexer )
          weights[featureIndexer.getIndex(f)] =
                  (that.featureIndexer.contains(f) ? that.weights[that.featureIndexer.getIndex(f)] : 0.);
      }
    } else {
      throw new IllegalArgumentException();
    }
  }
  @Override
  public Params merge(Params that_) {
    if(that_ instanceof  ParamsVec) {
      ParamsVec that = (ParamsVec) that_;
      // Join the two featureIndexers
      Indexer<Feature> joinedIndex = new Indexer<>();
      joinedIndex.addAll(featureIndexer);
      joinedIndex.addAll(that.featureIndexer);
      ParamsVec ret = new ParamsVec(K, joinedIndex);
      for( Feature f : featureIndexer )
        ret.weights[ret.featureIndexer.getIndex(f)] += weights[featureIndexer.getIndex(f)];
      for( Feature f : that.featureIndexer )
        ret.weights[ret.featureIndexer.getIndex(f)] += that.weights[that.featureIndexer.getIndex(f)];
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

  public void initRandom(Random random, double noise) {
    for (int j = 0; j < numFeatures; j++)
      weights[j] = noise * (2 * random.nextDouble() - 1);
  }

  public void plusEquals(double scale, Params that_) {
    if(that_ instanceof ParamsVec) {
      ParamsVec that = (ParamsVec) that_;
      if(that.featureIndexer == featureIndexer) {
        // They share the same featureIndexer, so lets optimize
        for(int i = 0; i < weights.length; i++)
          this.weights[i] += scale * that.weights[i];
      } else {
        for( Feature f : featureIndexer )
          if( that.featureIndexer.contains(f) )
            this.weights[featureIndexer.getIndex(f)] += scale * that.weights[that.featureIndexer.getIndex(f)];
      }
    } else {
      throw new IllegalArgumentException();
    }
  }

  public double dot(Params that_) {
    if(that_ instanceof ParamsVec) {
      ParamsVec that = (ParamsVec) that_;
      double prod = 0.;
      if(that.featureIndexer == featureIndexer) {
        for(int i = 0; i < weights.length; i++)
          prod += this.weights[i] * that.weights[i];
      } else {
        for( Feature f : featureIndexer )
          if( that.featureIndexer.contains(f) )
            prod += this.weights[featureIndexer.getIndex(f)] * that.weights[that.featureIndexer.getIndex(f)];
      }

      return prod;
    } else {
      throw new IllegalArgumentException();
    }
  }

  /**
   * Create a copy of that_ where ever they are equal.
   * @param that_ - copy
   * @return
   */
  public Params restrict(Params that_) {
    if(that_ instanceof ParamsVec) {
      ParamsVec that = (ParamsVec) that_;
      ParamsVec ret = new ParamsVec(K, featureIndexer);
      ret.copyOver(that);
      return ret;
    } else {
      throw new IllegalArgumentException();
    }
  }

  public void scaleEquals(double scale) {
    for(int i = 0; i < weights.length; i++)
      weights[i] *= scale;
  }

  public Params scale(double scale) {
    ParamsVec other = new ParamsVec(this);
    other.scaleEquals(scale);
    return other;
  }

  // -- ParamsVec specific
  public double get(Feature f) {
    return this.weights[featureIndexer.getIndex(f)];
  }
  public void set(Feature f, double val) {
    this.weights[featureIndexer.getIndex(f)] = val;
  }
  public void incr(Feature f, double val) {
    this.weights[featureIndexer.getIndex(f)] += val;
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
//      vec3.weights[vec3.featureIndexer.getIndex(f)] =
//              (vec1.featureIndexer.contains(f) ? vec1.weights[vec1.featureIndexer.getIndex(f)] : 0.)
//                      + (vec2.featureIndexer.contains(f) ? vec2.weights[vec2.featureIndexer.getIndex(f)] : 0.);
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
//      vec3.weights[vec3.featureIndexer.getIndex(f)] =
//              (vec1.featureIndexer.contains(f) ? vec1.weights[vec1.featureIndexer.getIndex(f)] : 0.)
//                      - (vec2.featureIndexer.contains(f) ? vec2.weights[vec2.featureIndexer.getIndex(f)] : 0.);
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
