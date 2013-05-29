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
  // L gives the length of the observation sequence.
  public Hypergraph<Example> createHypergraph(int L, double[] params, double[] counts, double increment) {
    return createHypergraph( L, null, params, counts, increment );
  }
  public Hypergraph<Example> createHypergraph(Example ex, double[] params, double[] counts, double increment) {
    return createHypergraph( ex.x.length, ex, params, counts, increment );
  }
  abstract Hypergraph<Example> createHypergraph(int L, Example ex, double[] params, double[] counts, double increment);

  Hypergraph.LogHyperedgeInfo<Example> nullInfo = new Hypergraph.LogHyperedgeInfo<Example>() {
    public double getLogWeight() { return 0; }
    public void setPosterior(double prob) { }
    public Example choose(Example ex) { return ex; }
  };

  Hypergraph.LogHyperedgeInfo<Example> hiddenEdgeInfo(final int j, final int v) {
    return new Hypergraph.LogHyperedgeInfo<Example>() {
      public double getLogWeight() { return 0; }
      public void setPosterior(double prob) { }
      public Example choose(Example ex) {
        ex.h[j] = v;
        return ex;
      }
      public String toString() { return "edge " + j + " " + v; }
    };
  }
  Hypergraph.LogHyperedgeInfo<Example> debugEdge(final String msg) {
    return new Hypergraph.LogHyperedgeInfo<Example>() {
      public double getLogWeight() { return 0; }
      public void setPosterior(double prob) { }
      public Example choose(Example ex) { return ex; }
      public String toString() { return msg; }
    };
  }

  Hypergraph.LogHyperedgeInfo<Example> hiddenEdgeInfo(double[] params, double[] counts, int f, double increment, final int j, final int v) {
    return new Hypergraph.MultinomialLogHyperedgeInfo<Example>(params, counts, f, increment) {
      public Example choose(Example ex) {
        ex.h[j] = v;
        return ex;
      }
    };
  }

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

