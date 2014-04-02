package learning.models.loglinear;

import java.util.*;

import fig.basic.*;
import learning.models.ExponentialFamilyModel;
import learning.models.Params;
import learning.common.Counter;

public abstract class Model extends ExponentialFamilyModel<Example> {
  public int K;
  public int getK() { return K; }
  public int D;
  public int getD() { return D; }
  public int L; // Number of 'views' or length.
  public int getL() { return L; }
  public final Indexer<Feature> featureIndexer = new Indexer<>();
  public ParamsVec fullParams;
//  public final Indexer<Feature> restrictedFeatureIndexer = new Indexer<>();
  public int numFeatures() { return featureIndexer.size(); }
  public ParamsVec newParams() {
//    return new ParamsVec(K, restrictedFeatureIndexer);
    return new ParamsVec(K, featureIndexer);
  }

  abstract Example newExample();
  abstract Example newExample(int[] x);
  // L gives the length of the observation sequence.
  // Use the ExponentialFamilyModel interface
  @Deprecated
  public Hypergraph<Example> createHypergraph(double[] params, double[] counts, double increment) {
    return createHypergraph( null, params, counts, increment );
  }
  @Deprecated
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

  @Override
  public double getLogLikelihood(Params params, int L) {
    int temp = this.L;
    this.L = L;
    double lhood = getLogLikelihood(params);
    this.L = temp;
    return lhood;
  }
  @Override
  public double getLogLikelihood(Params params, Example example) {
//    ParamsVec parameters = (ParamsVec) params;
    fullParams.copyOver(params);
    Hypergraph<Example> Hp = createHypergraph(example, fullParams.weights, null, 0.);
    Hp.computePosteriors(false);
    return Hp.getLogZ();
  }

  @Override
  public void updateMarginals(Params params, int L, double scale, double count, Params marginals_) {
    int temp = this.L;
    this.L = L;
    updateMarginals(params, (Example) null, scale, count, marginals_);
    this.L = temp;
  }
  @Override
  public void updateMarginals(Params params, Example example, double scale, double count, Params marginals_) {
    fullParams.copyOver(params);
//    ParamsVec parameters = (ParamsVec) params;
//    ParamsVec marginals = (ParamsVec) marginals_;
    ParamsVec marginals = (ParamsVec) fullParams.newParams();
    Hypergraph<Example> Hp = createHypergraph(example, fullParams.weights, marginals.weights, scale);
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);
    marginals_.plusEquals(marginals);
  }
  @Override
  public Counter<Example> drawSamples(Params params, Random rnd, int n) {
    fullParams.copyOver(params);
//    ParamsVec parameters = (ParamsVec) params;
    ParamsVec counts = (ParamsVec) fullParams.newParams();
    Hypergraph<Example> Hp = createHypergraph(fullParams.weights, counts.weights, 1);
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

