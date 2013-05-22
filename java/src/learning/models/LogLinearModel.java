package learning.models;
import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

/**
Conventions:
 H: hypergraph (q is target distribution, p is predicted distribution)

Notes:
 - Measure error using L1 distance on mean parameters (counts), not on the parameters.
*/

interface Feature {
}

// Conjunction of latent state and some description (for node potentials in HMM).
class UnaryFeature implements Feature {
  final int h;  // Latent state associated with this feature
  final String description;
  UnaryFeature(int h, String description) {
    this.h = h;
    this.description = description;
  }
  @Override public String toString() { return "h="+h+":"+description; }
  @Override public boolean equals(Object _that) {
    if (!(_that instanceof UnaryFeature)) return false;
    UnaryFeature that = (UnaryFeature)_that;
    return this.h == that.h && this.description.equals(that.description);
  }
  @Override public int hashCode() { return h * 37 + description.hashCode(); }
}

// Conjunction of two latent states (for edge potentials in HMM).
class BinaryFeature implements Feature {
  final int h1, h2;
  BinaryFeature(int h1, int h2) { this.h1 = h1; this.h2 = h2; }
  @Override public String toString() { return "h1="+h1+",h2="+h2; }
  @Override public boolean equals(Object _that) {
    if (!(_that instanceof BinaryFeature)) return false;
    BinaryFeature that = (BinaryFeature)_that;
    return this.h1 == that.h1 && this.h2 == that.h2;
  }
  @Override public int hashCode() { return h1 * 37 + h2; }
}

abstract class Model {
  Indexer<Feature> featureIndexer;

  abstract Example newExample();
  abstract Hypergraph createHypergraph(Example ex, double[] params, double[] counts, double increment);

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

class Example {
  Hypergraph Hq;  // For inference conditioned on the observations (represents q(h|x)).
  int[] x;  // Values of observed nodes
}

class MixtureModel extends Model {
  int K, L, D;

  public Example newExample() {
    Example ex = new Example();
    ex.x = new int[L];
    return ex;
  }

  public Hypergraph createHypergraph(Example ex, double[] params, double[] counts, double increment) {
    Hypergraph H = new Hypergraph();
    //H.debug = true;
    Object rootNode = H.sumStartNode();
    for (int h = 0; h < K; h++) {  // For each value of hidden states...
      String hNode = "h="+h;
      H.addProdNode(hNode);
      H.addEdge(rootNode, hNode);
      for (int j = 0; j < L; j++) {  // For each view j...
        String xNode = "h="+h+",x"+j;
        if (ex != null) {  // Numerator: generate x[j]
          int f = featureIndexer.getIndex(new UnaryFeature(h, "x="+ex.x[j]));
          if (params != null)
            H.addEdge(hNode, H.endNode, edgeInfo(params, counts, f, increment));
        } else {  // Denominator: generate each possible assignment x[j] = a
          H.addSumNode(xNode);
          H.addEdge(hNode, xNode);
          for (int a = 0; a < D; a++) {
            int f = featureIndexer.getIndex(new UnaryFeature(h, "x="+a));
            if (params != null)
              H.addEdge(xNode, H.endNode, edgeInfo(params, counts, f, increment, j, a));
          }
        }
      }
    }
    return H;
  }
}

class LikelihoodFunctionState implements Maximizer.FunctionState {
  // Set via initialization.
  Model model;
  List<Example> examples;
  int K;
  Indexer<Feature> featureIndexer;
  int numFeatures;
  boolean storeHypergraphs;
  double regularization;

  boolean valid = false;
  double objective;
  ParamsVec params;  // Canonical parameters
  ParamsVec gradient;
  ParamsVec pCounts;  // Mean parameters
  Hypergraph Hp;

  void init(Random initRandom, double initNoise) {
    params = new ParamsVec(K, featureIndexer);
    gradient = new ParamsVec(K, featureIndexer);
    pCounts = new ParamsVec(K, featureIndexer);
    Hp = model.createHypergraph(null, params.weights, pCounts.weights, 1);
    params.initRandom(initRandom, initNoise);
  }

  public void invalidate() { valid = false; }
  public double[] point() { return params.weights; }
  public double value() { compute(); return objective; }
  public double[] gradient() { compute(); return gradient.weights; }

  public void compute() {
    if (valid) return;
    valid = true;
    objective = 0;
    gradient.clear();
    pCounts.clear();

    for (Example ex : examples) {
      // Numerator
      Hypergraph Hq = ex.Hq;
      if (Hq == null) Hq = model.createHypergraph(ex, params.weights, gradient.weights, 1.0/examples.size());
      if (storeHypergraphs) ex.Hq = Hq;

      Hq.computePosteriors(false);
      Hq.fetchPosteriors(false);
      objective += Hq.getLogZ() * 1.0/examples.size();
    }

    // Denominator (globally normalized for now)
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);
    ListUtils.incr(gradient.weights, -1, pCounts.weights);
    objective -= Hp.getLogZ();

    // Regularization
    if (regularization > 0) {
      for (int f = 0; f < numFeatures; f++) {
        gradient.weights[f] -= regularization * params.weights[f];
        objective -= 0.5 * regularization * params.weights[f] * params.weights[f];
      }
    }
  }
}

class ParamsVec {
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

/**
 * Perform learning of various log-linear models.
 * Assume globally normalized models with L-BFGS optimization.
 * For NIPS 2013.
 */
public class LogLinearModel implements Runnable {
  public static class Options {
    public enum ModelType { mixture, hmm, tallMixture, grid, factMixture };

    @Option(gloss="Type of model") public ModelType modelType = ModelType.mixture;
    @Option(gloss="Number of values of the hidden variable") public int K = 3;
    @Option(gloss="Number of possible values of output") public int D = 5;
    @Option(gloss="Length of observation sequence") public int L = 4;
    @Option(gloss="Random seed for initialization") public Random initRandom = new Random(1);
    @Option(gloss="Random seed for generating artificial data") public Random genRandom = new Random(1);
    @Option(gloss="Random seed for the true model") public Random trueParamsRandom = new Random(1);
    @Option(gloss="Number of optimization iterations") public int numIters = 100;
    @Option(gloss="Number of examples to generate") public int genNumExamples = 100;
    @Option(gloss="Whether to keep hypergraphs for all the examples") public boolean storeHypergraphs = true;
    @Option(gloss="How much variation in true parameters") public double trueParamsNoise = 1;
    @Option(gloss="How much variation in initial parameters") public double initParamsNoise = 0.01;
    @Option(gloss="Regularization") public double regularization = 0;
    @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
    @OptionSet(name="backtrack") public BacktrackingLineSearch.Options backtrack = new BacktrackingLineSearch.Options();
  }

  public static Options opts = new Options();

  Indexer<Feature> featureIndexer = new Indexer<Feature>();
  int numFeatures() { return featureIndexer.size(); }

  Model model;
  ParamsVec trueParams;  // True parameters that we're trying to learn
  ParamsVec trueCounts;
  List<Example> examples = new ArrayList<Example>();  // Examples generated from the true model

  void createExamples() {
    // Create the true parameters
    trueParams = new ParamsVec(opts.K, featureIndexer);
    trueParams.initRandom(opts.trueParamsRandom, 1);
    trueCounts = new ParamsVec(opts.K, featureIndexer);
    Hypergraph Hp = model.createHypergraph(null, trueParams.weights, trueCounts.weights, 1);
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);
    trueParams.write(Execution.getFile("true.params"));
    trueCounts.write(Execution.getFile("true.counts"));

    // Generate examples from the model
    for (int i = 0; i < opts.genNumExamples; i++) {
      Example ex = model.newExample();
      Hp.fetchSampleHyperpath(opts.genRandom, ex);
      examples.add(ex);
      //LogInfo.logs("x = %s", Fmt.D(ex.x));
    }
    logs("Generated %d examples", examples.size());
  }

  public void initModel() {
    switch (opts.modelType) {
      case mixture:
        MixtureModel model = new MixtureModel();
        model.featureIndexer = featureIndexer;
        model.K = opts.K;
        model.L = opts.L;
        model.D = opts.D;
        this.model = model;
        break;
      default:
        throw new RuntimeException("Unhandled model type: " + opts.modelType);
    }

    // Run once to just instantiate features
    model.createHypergraph(null, null, null, 0);
  }

  String logStat(String key, Object value) {
    LogInfo.logs("%s = %s", key, value);
    Execution.putOutput(key, value);
    return key+"="+value;
  }

  public void run() {
    PrintWriter eventsOut = IOUtils.openOutHard(Execution.getFile("events"));

    initModel();
    createExamples();

    // Optimize
    LBFGSMaximizer maximizer = new LBFGSMaximizer(opts.backtrack, opts.lbfgs);
    LikelihoodFunctionState state = new LikelihoodFunctionState();
    state.model = model;
    state.examples = examples;
    state.K = opts.K;
    state.featureIndexer = featureIndexer;
    state.numFeatures = numFeatures();
    state.storeHypergraphs = opts.storeHypergraphs;
    state.regularization = opts.regularization;
    state.init(opts.initRandom, opts.initParamsNoise);
    int[] perm = new int[opts.K];
    for (int iter = 0; iter < opts.numIters; iter++) {
      LogInfo.begin_track("Iteration %d/%d", iter, opts.numIters);
      List<String> items = new ArrayList<String>();
      items.add("iter="+iter);
      items.add(logStat("objective", state.value()));
      items.add(logStat("paramsError", state.params.computeDiff(trueParams, perm)));
      items.add(logStat("paramsPerm", Fmt.D(perm)));
      items.add(logStat("countsError", state.pCounts.computeDiff(trueCounts, perm)));
      items.add(logStat("countsPerm", Fmt.D(perm)));
      eventsOut.println(StrUtils.join(items, "\t"));
      eventsOut.flush();
      if (maximizer.takeStep(state)) { LogInfo.end_track(); break; }
      LogInfo.end_track();
    }
    state.params.write(Execution.getFile("params"));
    state.pCounts.write(Execution.getFile("counts"));
  }

  public static void main(String[] args) {
    Execution.run(args, new LogLinearModel(), "Main", opts);
  }
}
