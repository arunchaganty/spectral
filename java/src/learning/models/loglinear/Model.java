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
  public int L;
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

  class UpdatingEdgeInfo implements Hypergraph.LogHyperedgeInfo<Example> {
    int j;
    int v;
    boolean updateHidden;

    public double getLogWeight() { return 0; }
    public void setPosterior(double prob) { }

    UpdatingEdgeInfo(final int j, final int v, boolean updateHidden) {
      this.j = j;
      this.v = v;
      this.updateHidden = updateHidden;
    }
    public Example choose(Example ex) {
      if( updateHidden )
        ex.h[j] = v;
      else
        ex.x[j] = v;
      return ex;
    }
    public String toString() { return "edge " + j + " " + v; }
  }
  class UpdatingMultinomialEdgeInfo extends Hypergraph.MultinomialLogHyperedgeInfo<Example> {
    int j;
    int v;
    boolean updateHidden;
    UpdatingMultinomialEdgeInfo(double[] params, double[] counts, int f, double increment, final int j, final int v, boolean updateHidden) {
      super(params, counts, f, increment);
      this.j = j;
      this.v = v;
      this.updateHidden = updateHidden;
    }
    public Example choose(Example ex) {
      if( updateHidden )
        ex.h[j] = v;
      else
        ex.x[j] = v;
      return ex;
    }
  }


  Hypergraph.LogHyperedgeInfo<Example> hiddenEdgeInfo(final int j, final int v) {
    return new UpdatingEdgeInfo(j, v, true);
  }

  Hypergraph.LogHyperedgeInfo<Example> hiddenEdgeInfo(double[] params, double[] counts, int f, double increment, final int j, final int v) {
    return new UpdatingMultinomialEdgeInfo(params, counts, f, increment, j, v, true);
  }
  Hypergraph.LogHyperedgeInfo<Example> edgeInfo(double[] params, double[] counts, int f, double increment) {
    return new Hypergraph.MultinomialLogHyperedgeInfo<Example>(params, counts, f, increment);
  }
  Hypergraph.LogHyperedgeInfo<Example> edgeInfo(double[] params, double[] counts, int f, double increment, final int j, final int v) {
    return new UpdatingMultinomialEdgeInfo(params, counts, f, increment, j, v, false);
  }
}

