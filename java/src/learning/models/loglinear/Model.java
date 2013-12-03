package learning.models.loglinear;

import java.io.*;
import java.util.*;

import learning.linalg.MatrixOps;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import learning.models.ExponentialFamilyModel;
import learning.utils.Counter;

import static fig.basic.LogInfo.*;

public abstract class Model extends ExponentialFamilyModel<Example> {
  public int K;
  public int getK() { return K; }
  public int D;
  public int getD() { return D; }
  public int L; // Number of 'views' or length.
  public Indexer<Feature> featureIndexer = new Indexer<Feature>();
  public int numFeatures() { return featureIndexer.size(); }
  public ParamsVec newParamsVec() { return new ParamsVec(K, featureIndexer); }

  abstract Example newExample();
  abstract Example newExample(int[] x);
  // L gives the length of the observation sequence.
  public Hypergraph<Example> createHypergraph(double[] params, double[] counts, double increment) {
    return createHypergraph( null, params, counts, increment );
  }
  abstract Hypergraph<Example> createHypergraph(Example ex, double[] params, double[] counts, double increment);
  @Deprecated
  public Hypergraph<Example> createHypergraph(int L, Example ex, double[] params, double[] counts, double increment) {
    return createHypergraph( ex, params, counts, increment );
  }

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

  public double getLogLikelihood(ParamsVec parameters) {
    Hypergraph<Example> Hp = createHypergraph(parameters.weights, null, 0.);
    Hp.computePosteriors(false);
    return Hp.getLogZ();
  }
  public double getLogLikelihood(ParamsVec parameters, Example example) {
    Hypergraph<Example> Hp = createHypergraph(example, parameters.weights, null, 0.);
    Hp.computePosteriors(false);
    return Hp.getLogZ();
  }
  public double getLogLikelihood(ParamsVec parameters, Counter<Example> examples) {
    double lhood = 0.;
    for(Example example: examples) {
      Hypergraph<Example> Hp = createHypergraph(example, parameters.weights, null, 0.);
      Hp.computePosteriors(false);
      lhood += examples.getFraction(example) * Hp.getLogZ();
    }
    return lhood;
  }
  public ParamsVec getMarginals(ParamsVec parameters) {
    ParamsVec counts = newParamsVec();
    Hypergraph<Example> Hp = createHypergraph(parameters.weights, counts.weights, 1.);
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);
    return counts;
  }
  public ParamsVec getMarginals(ParamsVec parameters, Example example) {
    ParamsVec counts = newParamsVec();
    Hypergraph<Example> Hp = createHypergraph(example, parameters.weights, counts.weights, 1.);
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);
    return counts;
  }
  public ParamsVec getMarginals(ParamsVec parameters, Counter<Example> examples) {
    ParamsVec counts = newParamsVec();
    for(Example example: examples) {
      Hypergraph<Example> Hp = createHypergraph(example, parameters.weights, counts.weights, examples.getFraction(example));
      Hp.computePosteriors(false);
      Hp.fetchPosteriors(false);
    }
    return counts;
  }
  public Counter<Example> drawSamples(ParamsVec parameters, Random rnd, int n) {
    ParamsVec counts = newParamsVec();
    Hypergraph<Example> Hp = createHypergraph(parameters.weights, counts.weights, 1);
    // Necessary preprocessing before you can generate hyperpaths
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);

    Counter<Example> examples = new Counter<>();
    for (int i = 0; i < n; i++) {
      Example ex = newExample();
      Hp.fetchSampleHyperpath(rnd, ex);
      examples.add(ex);
    }
    return examples;
  }

  /**
   * Choose idx
   */
  void generateExamples(Example current, int idx, List<Example> examples) {
    if( idx == current.x.length ) {
      examples.add(new Example(current.x));
    } else {
      // Make a choice for this index
      current.x[idx] = 0;
      generateExamples(current, idx+1, examples);
      current.x[idx] = 1;
      generateExamples(current, idx+1, examples);
    }
  }
  List<Example> generateExamples(int L) {
    List<Example> examples = new ArrayList<>((int)Math.pow(2,L));
    Example ex = new Example(new int[L]);
    generateExamples(ex, 0, examples);
    return examples;
  }


}

