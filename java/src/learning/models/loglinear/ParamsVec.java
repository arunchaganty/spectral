package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

public class ParamsVec {
  int K;  // Number of hidden states
  Indexer<Feature> featureIndexer;
  int numFeatures;
  ParamsVec(int K, Indexer<Feature> featureIndexer) {
    this.K = K;
    this.featureIndexer = featureIndexer;
    this.numFeatures = featureIndexer.size();
    this.weights = new double[numFeatures];
  }

  double[] weights;

  void initRandom(Random random, double noise) {
    for (int j = 0; j < numFeatures; j++)
      weights[j] = noise * (2 * random.nextDouble() - 1);
  }

  void clear() { ListUtils.set(weights, 0); }
  void incr(double scale, ParamsVec that) { ListUtils.incr(this.weights, scale, that.weights); }
  double dot(ParamsVec that) { return ListUtils.dot(this.weights, that.weights); }

  double computeDiff(ParamsVec that, int[] perm) {
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

  void write(String path) {
    PrintWriter out = IOUtils.openOutHard(path);
    //for (int f : ListUtils.sortedIndices(weights, true))
    for (int f = 0; f < numFeatures; f++)
      out.println(featureIndexer.getObject(f) + "\t" + weights[f]);
    out.close();
  }
}
