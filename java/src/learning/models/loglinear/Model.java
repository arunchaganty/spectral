package learning.models.loglinear;

import java.util.*;

import fig.basic.*;
import learning.models.ExponentialFamilyModel;
import learning.models.Params;
import learning.utils.Counter;

public abstract class Model extends ExponentialFamilyModel<Example> {
  public int K;
  public int getK() { return K; }
  public int D;
  public int getD() { return D; }
  public int L; // Number of 'views' or length.
  public Indexer<Feature> featureIndexer = new Indexer<Feature>();
  public int numFeatures() { return featureIndexer.size(); }
  public ParamsVec newParams() { return new ParamsVec(K, featureIndexer); }

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

  public double getLogLikelihood(Params params) {
    ParamsVec parameters = (ParamsVec) params;
    Hypergraph<Example> Hp = createHypergraph(parameters.weights, null, 0.);
    Hp.computePosteriors(false);
    return Hp.getLogZ();
  }
  public double getLogLikelihood(Params params, Example example) {
    ParamsVec parameters = (ParamsVec) params;
    Hypergraph<Example> Hp = createHypergraph(example, parameters.weights, null, 0.);
    Hp.computePosteriors(false);
    return Hp.getLogZ();
  }
  public double getLogLikelihood(Params params, Counter<Example> examples) {
    ParamsVec parameters = (ParamsVec) params;
    double lhood = 0.;
    for(Example example: examples) {
      Hypergraph<Example> Hp = createHypergraph(example, parameters.weights, null, 0.);
      Hp.computePosteriors(false);
      lhood += examples.getFraction(example) * Hp.getLogZ();
    }
    return lhood;
  }
  @Override
  public void updateMarginals(Params params, Example example, double scale, Params marginals_) {
    ParamsVec parameters = (ParamsVec) params;
    ParamsVec marginals = (ParamsVec) marginals_;
    Hypergraph<Example> Hp = createHypergraph(example, parameters.weights, marginals.weights, scale);
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);
  }
  @Override
  public Counter<Example> drawSamples(Params params, Random rnd, int n) {
    ParamsVec parameters = (ParamsVec) params;
    ParamsVec counts = newParams();
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
      for(int i = 0; i < D; i++) {
        current.x[idx] = i;
        generateExamples(current, idx+1, examples);
      }
    }
  }
  List<Example> generateExamples(int L) {
    List<Example> examples = new ArrayList<>((int)Math.pow(2,L));
    Example ex = new Example(new int[L]);
    generateExamples(ex, 0, examples);
    return examples;
  }

}

