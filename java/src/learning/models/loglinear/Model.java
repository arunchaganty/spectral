package learning.models.loglinear;

import java.io.*;
import java.util.*;

import learning.linalg.MatrixOps;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

public abstract class Model {
  public int K;  // Number of latent states
  public int D;  // Number of emissions
  public Indexer<Feature> featureIndexer = new Indexer<Feature>();
  public int numFeatures() { return featureIndexer.size(); }

  ParamsVec newParamsVec() { return new ParamsVec(K, featureIndexer); }

  abstract Example newExample();
  abstract Example newExample(int[] x);
  abstract Hypergraph<Example> createHypergraph(Example ex, double[] params, double[] counts, double increment);

  Hypergraph.LogHyperedgeInfo<Example> nullInfo = new Hypergraph.LogHyperedgeInfo<Example>() {
    public double getLogWeight() { return 0; }
    public void setPosterior(double prob) { }
    public Example choose(Example ex) { return ex; }
  };

  Hypergraph.LogHyperedgeInfo<Example> edgeInfo(double[] params, double[] counts, int f, double increment) {
    return new Hypergraph.MultinomialLogHyperedgeInfo<Example>(params, counts, f, increment);
  }
  Hypergraph.LogHyperedgeInfo<Example> edgeInfo(double[] params, double[] counts, int f, double increment, final int j, final int v) {
    return new Hypergraph.MultinomialLogHyperedgeInfo<Example>(params, counts, f, increment) {
      public Example choose(Example ex) {
        ex.x[j] = v;
        return ex;
      }
    };
  }
}

