package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

class Example {
  Hypergraph Hq;  // For inference conditioned on the observations (represents q(h|x)).
  int[] x;  // Values of observed nodes
}

// A term in the objective function.
abstract class ObjectiveTerm {
  double value;
  ParamsVec counts;

  // Populate |value| and |counts| based on the current state.
  public abstract void infer(boolean needGradient);
}

class ZeroTerm extends ObjectiveTerm {
  public ZeroTerm(ParamsVec counts) { this.counts = counts; }
  public void infer(boolean needGradient) { }
}

// Linear function with given coefficients
class LinearTerm extends ObjectiveTerm {
  ParamsVec params;
  ParamsVec coeffs;
  boolean[] measuredFeatures;  // Which features to include

  public LinearTerm(ParamsVec params, ParamsVec coeffs, boolean[] measuredFeatures) {
    this.params = params;
    this.coeffs = coeffs;
    this.measuredFeatures = measuredFeatures;
  }

  public void infer(boolean needGradient) {
    value = 0;
    for (int j = 0; j < measuredFeatures.length; j++) {
      if (!measuredFeatures[j]) continue;
      value += params.weights[j] * coeffs.weights[j];
    }
    if (needGradient) this.counts = coeffs;
  }
}

// Log-partition function over all posible h, summed over the given x's
class ExamplesTerm extends ObjectiveTerm {
  Model model;
  ParamsVec params;
  List<Example> examples;
  boolean storeHypergraphs;

  ExamplesTerm(Model model, ParamsVec params, ParamsVec counts, List<Example> examples, boolean storeHypergraphs) {
    this.model = model;
    this.params = params;
    this.counts = counts;
    this.examples = examples;
    this.storeHypergraphs = storeHypergraphs;
  }

  public void infer(boolean needGradient) {
    logs("ExamplesTerm.infer");
    value = 0;
    if (needGradient) counts.clear();
    for (Example ex : examples) {
      Hypergraph Hq = ex.Hq;
      if (Hq == null) Hq = model.createHypergraph(ex, params.weights, counts.weights, 1.0/examples.size());
      if (storeHypergraphs) ex.Hq = Hq;

      Hq.computePosteriors(false);
      if (needGradient) Hq.fetchPosteriors(false);
      value += Hq.getLogZ() * 1.0/examples.size();
    }
  }
}

// Log-partition function over all possible (x,h)
class GlobalTerm extends ObjectiveTerm {
  Model model;
  ParamsVec params;
  Hypergraph H;

  public GlobalTerm(Model model, ParamsVec params, ParamsVec counts) {
    this.model = model;
    this.params = params;
    this.counts = counts;
    H = model.createHypergraph(null, params.weights, counts.weights, 1);
  }

  public void infer(boolean needGradient) {
    if (needGradient) counts.clear();
    H.computePosteriors(false);
    if (needGradient) H.fetchPosteriors(false);
    value = H.getLogZ();
  }
}

// Objective function: target - pred.
//  - Model features: \phi (canonical parameters \theta, mean parameters \mu)
//  - Measurement features: \sigma (canonical parameters \beta, mean parameters \tau)
//    (assume \sigma \subset \phi)
// Looking at the gradient:
//  - target is one of the following:
//    * Concrete sufficient statistics \mu, \tau [targetStats]
//    * Examples \E_{p(h|x; \theta)}[\phi(x,h)] [targetExamples]
//  - pred is one the following:
//    * Examples \E_{p(h|x; \theta)}[\sigma(x,h)] [predExamples]
//    * Model \E_{p(x,h; \theta)}[\phi(x,h)] [nothing]
// In the implementation, we do everything with all the model features \phi,
// but just zero out the gradient for the non-active subset if we want \sigma.
class LikelihoodFunctionState implements Maximizer.FunctionState {
  Model model;
  ObjectiveTerm target, pred;
  boolean[] measuredFeatures;

  // TODO: generalize to arbitrary quadratic (A w - b)^2
  double regularization;

  // Optimization state
  boolean objectiveValid = false;
  boolean gradientValid = false;
  double objective;
  ParamsVec params;  // Canonical parameters
  ParamsVec gradient;  // targetCounts - predCounts - \nabla regularization
  Hypergraph Hp;

  public LikelihoodFunctionState(Model model, ParamsVec params, ObjectiveTerm target, ObjectiveTerm pred,
      boolean[] measuredFeatures, double regularization, Random initRandom, double initNoise) {
    this.model = model;
    this.params = params;
    this.target = target;
    this.pred = pred;
    this.measuredFeatures = measuredFeatures;
    this.regularization = regularization;

    // Create state
    this.gradient = model.newParamsVec();
    this.params.initRandom(initRandom, initNoise);
  }

  public void invalidate() { objectiveValid = gradientValid = false; }
  public double[] point() { return params.weights; }
  public double value() { compute(false); return objective; }
  public double[] gradient() { compute(true); return gradient.weights; }

  public void compute(boolean needGradient) {
    if (needGradient ? gradientValid : objectiveValid) return;
    objectiveValid = true;

    target.infer(needGradient);
    pred.infer(needGradient);

    // Compute objective value
    objective = target.value - pred.value;
    if (regularization > 0) {
      for (int j = 0; j < model.numFeatures(); j++) {
        if (!measuredFeatures[j]) continue;
        objective -= 0.5 * regularization * params.weights[j] * params.weights[j];
      }
    }

    // Compute gradient (if needed)
    if (needGradient) {
      gradientValid = true;
      gradient.clear();
      //logs("objective = %s, gradient (%s): [%s] - [%s] - %s [%s]", Fmt.D(objective), Fmt.D(measuredFeatures), Fmt.D(target.counts.weights), Fmt.D(pred.counts.weights), regularization, Fmt.D(params.weights));
      for (int j = 0; j < model.numFeatures(); j++) {
        if (!measuredFeatures[j]) continue;
        gradient.weights[j] += target.counts.weights[j] - pred.counts.weights[j];

        // Regularization
        if (regularization > 0)
          gradient.weights[j] -= regularization * params.weights[j];
      }
    }
  }
}

/**
 * Perform learning of various log-linear models.
 * Assume globally normalized models with L-BFGS optimization.
 * For NIPS 2013.
 */
public class Main implements Runnable {
  public static class Options {
    public enum ModelType { mixture, hmm, tallMixture, grid, factMixture };
    public enum ObjectiveType { supervised, measurements, unsupervised_em, unsupervised_gradient };

    @Option(gloss="Type of model") public ModelType modelType = ModelType.mixture;
    @Option(gloss="Number of values of the hidden variable") public int K = 3;
    @Option(gloss="Number of possible values of output") public int D = 5;
    @Option(gloss="Length of observation sequence") public int L = 3;
    @Option(gloss="Random seed for initialization") public Random initRandom = new Random(1);
    @Option(gloss="Random seed for generating artificial data") public Random genRandom = new Random(1);
    @Option(gloss="Random seed for the true model") public Random trueParamsRandom = new Random(1);
    @Option(gloss="Number of optimization outside iterations") public int numIters = 100;
    @Option(gloss="Number of optimization inside E-step iterations") public int eNumIters = 1;
    @Option(gloss="Number of optimization inside M-step iterations") public int mNumIters = 1;
    @Option(gloss="Number of examples to generate") public int genNumExamples = 100;
    @Option(gloss="Whether to keep hypergraphs for all the examples") public boolean storeHypergraphs = true;
    @Option(gloss="How much variation in true parameters") public double trueParamsNoise = 1;
    @Option(gloss="How much variation in initial parameters") public double initParamsNoise = 0.01;
    @Option(gloss="Regularization for the E-step (important for inconsistent moments)") public double eRegularization = 1e-3;
    @Option(gloss="Regularization for the M-step") public double mRegularization = 0;

    @Option(gloss="Type of training to use") public ObjectiveType objectiveType = ObjectiveType.unsupervised_gradient;
    @Option(gloss="Type of optimization to use") public boolean useLBFGS = true;

    @Option(gloss="Use expected measurements (with respect to true distribution)") public boolean expectedMeasurements = true;
    @Option(gloss="Include each (true) measurement with this prob") public double measurementProb = 1;
    @Option(gloss="Number of iterations before switching to full unsupervised") public int numMeasurementIters = Integer.MAX_VALUE;

    @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
    @OptionSet(name="backtrack") public BacktrackingLineSearch.Options backtrack = new BacktrackingLineSearch.Options();
  }

  public static Options opts = new Options();

  PrintWriter eventsOut;
  Model model;
  ParamsVec trueParams;  // True parameters that we're trying to learn
  ParamsVec trueCounts;
  List<Example> examples = new ArrayList<Example>();  // Examples generated from the true model
  boolean[] measuredFeatures;  // Specifies which features are measured
  ParamsVec measurements;  // Estimated from method of moments (\tau)

  void generateExamples() {
    // Create the true parameters
    trueParams = model.newParamsVec();
    trueParams.initRandom(opts.trueParamsRandom, 1);
    trueCounts = model.newParamsVec();
    Hypergraph<Example> Hp = model.createHypergraph(null, trueParams.weights, trueCounts.weights, 1);
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
      case mixture: {
        MixtureModel model = new MixtureModel();
        model.L = opts.L;
        model.D = opts.D;
        this.model = model;
        break;
      }
      case hmm: {
        HiddenMarkovModel model = new HiddenMarkovModel();
        model.L = opts.L;
        model.D = opts.D;
        this.model = model;
        break;
      }
      case tallMixture: {
        TallMixture model = new TallMixture();
        model.L = opts.L;
        model.D = opts.D;
        this.model = model;
        break;
      }
      case grid: {
        Grid model = new Grid();
        model.width = 2;
        model.height = opts.L/2;
        model.L = opts.L;
        model.D = opts.D;
        this.model = model;
        break;
      }
      default:
        throw new RuntimeException("Unhandled model type: " + opts.modelType);
    }

    model.K = opts.K;

    // Run once to just instantiate features
    model.createHypergraph(null, null, null, 0);

    measuredFeatures = new boolean[model.numFeatures()];
  }

  String logStat(String key, Object value) {
    LogInfo.logs("%s = %s", key, value);
    Execution.putOutput(key, value);
    return key+"="+value;
  }

  void estimateMoments() {
    if (opts.objectiveType != Options.ObjectiveType.measurements) return;

    Random random = new Random(3);
    if (opts.expectedMeasurements) {
      measurements = model.newParamsVec();

      // Currently, using true measurements
      for (int j = 0; j < model.numFeatures(); j++) {
        measuredFeatures[j] = random.nextDouble() < opts.measurementProb;
        if (measuredFeatures[j])
          measurements.weights[j] = trueCounts.weights[j];
      }
    } else {
      // TODO: Implement via spectral.
      throw new RuntimeException("Not supported");
    }
  }

  Maximizer newMaximizer() {
    if (opts.useLBFGS) return new LBFGSMaximizer(opts.backtrack, opts.lbfgs);
    return new GradientMaximizer(opts.backtrack);
  }

  // Goal: min KL(q||p)
  void optimize() {
    ParamsVec eParams = model.newParamsVec();  // beta + theta (q)
    ParamsVec mParams = model.newParamsVec();  // theta (p)
    ParamsVec mCounts = model.newParamsVec();  // mu (deterministic function of mParams)

    boolean[] allMeasuredFeatures = new boolean[model.numFeatures()];
    for (int j = 0; j < model.numFeatures(); j++) allMeasuredFeatures[j] = true;

    ZeroTerm zeroTerm = new ZeroTerm(model.newParamsVec());
    LinearTerm measurementsTerm = new LinearTerm(eParams, measurements, measuredFeatures);  // \tau
    ExamplesTerm eExamplesTerm = new ExamplesTerm(model, eParams, model.newParamsVec(), examples, opts.storeHypergraphs);

    ExamplesTerm mExamplesTerm = new ExamplesTerm(model, mParams, model.newParamsVec(), examples, opts.storeHypergraphs);
    LinearTerm supervisedTerm = new LinearTerm(mParams, trueCounts, allMeasuredFeatures);  // Infinite data
    LinearTerm examplesOutTerm = new LinearTerm(mParams, eExamplesTerm.counts, allMeasuredFeatures);
    GlobalTerm globalTerm = new GlobalTerm(model, mParams, mCounts);

    // Construct the objective function
    ObjectiveTerm eTargetTerm, ePredTerm;
    ObjectiveTerm mTargetTerm, mPredTerm;
    switch (opts.objectiveType) {
      case supervised:
        eTargetTerm = zeroTerm;
        ePredTerm = zeroTerm;
        mTargetTerm = supervisedTerm;
        mPredTerm = globalTerm;
        break;
      case measurements:
        eTargetTerm = measurementsTerm;
        ePredTerm = eExamplesTerm;
        mTargetTerm = examplesOutTerm;
        mPredTerm = globalTerm;
        break;
      case unsupervised_em:
        eTargetTerm = zeroTerm;
        ePredTerm = eExamplesTerm;
        mTargetTerm = examplesOutTerm;
        mPredTerm = globalTerm;
        break;
      case unsupervised_gradient:
        eTargetTerm = zeroTerm;
        ePredTerm = zeroTerm;
        mTargetTerm = mExamplesTerm;
        mPredTerm = globalTerm;
        break;
      default:
        throw new RuntimeException("Invalid objectiveType: " + opts.objectiveType);
    }

    // For the E-step
    Maximizer eMaximizer = newMaximizer();
    LikelihoodFunctionState eState = new LikelihoodFunctionState(
        model, eParams, eTargetTerm, ePredTerm, measuredFeatures,
        opts.eRegularization, opts.initRandom, opts.initParamsNoise);

    // For the M-step
    Maximizer mMaximizer = newMaximizer();
    LikelihoodFunctionState mState = new LikelihoodFunctionState(
        model, mParams, mTargetTerm, mPredTerm, allMeasuredFeatures,
        opts.mRegularization, opts.initRandom, opts.initParamsNoise);

    boolean done = false;
    for (int iter = 0; iter < opts.numIters && !done; iter++) {
      ParamsVec old_mParams = model.newParamsVec();
      ListUtils.set(old_mParams.weights, mParams.weights);

      LogInfo.begin_track("Iteration %d/%d", iter, opts.numIters);
      boolean done1 = optimizeIter(iter, "E", eMaximizer, eState, opts.eNumIters);
      boolean done2 = optimizeIter(iter, "M", mMaximizer, mState, opts.mNumIters);
      done = done1 && done2;

      // Copy parameters from M-step to E-step.
      // This is the base measure for q in KL(q||p)
      if (ePredTerm == eExamplesTerm) {
        for (int j = 0; j < model.numFeatures(); j++)
          eState.params.weights[j] += mParams.weights[j] - old_mParams.weights[j];
      }

      // Dump out statistics
      // TODO: Implement a way of checking parameter error in a
      // block-wise way.
      int[] perm = new int[opts.K];
      List<String> items = new ArrayList<String>();
      items.add("iter="+iter);
      //items.add(logStat("paramsError", mParams.computeDiff(trueParams, perm)));
      //items.add(logStat("paramsPerm", Fmt.D(perm)));
      //items.add(logStat("countsError", mCounts.computeDiff(trueCounts, perm)));
      //items.add(logStat("countsPerm", Fmt.D(perm)));
      items.add(logStat("eObjective", eState.value()));
      items.add(logStat("mObjective", mState.value()));
      eventsOut.println(StrUtils.join(items, "\t"));
      eventsOut.flush();

      LogInfo.end_track();

      // Switch to pure unsupervised
      if (opts.objectiveType == Options.ObjectiveType.measurements && iter == opts.numMeasurementIters) {
        LogInfo.logs("Switching over from measurements to unsupervised_gradient");
        eTargetTerm = zeroTerm;
        ePredTerm = zeroTerm;
        mTargetTerm = mExamplesTerm;
        mPredTerm = globalTerm;
        eState = new LikelihoodFunctionState(
            model, eParams, eTargetTerm, ePredTerm, measuredFeatures,
            opts.eRegularization, opts.initRandom, opts.initParamsNoise);
        mState = new LikelihoodFunctionState(
            model, mParams, mTargetTerm, mPredTerm, allMeasuredFeatures,
            opts.mRegularization, opts.initRandom, opts.initParamsNoise);
      }
    }

    if (done) LogInfo.logs("Converged");
    mParams.write(Execution.getFile("params"));
    mCounts.write(Execution.getFile("counts"));
  }

  // Return whether already converged.
  boolean optimizeIter(int outerIter, String stepName, Maximizer maximizer, LikelihoodFunctionState state, int numIters) {
    LogInfo.begin_track("%s-step", stepName);
    state.invalidate();
    boolean done = false;
    // E-step
    int iter;
    for (iter = 0; iter < numIters && !done; iter++) {
      LogInfo.begin_track("Iteration %s/%s", iter, numIters);
      done = maximizer.takeStep(state);
      LogInfo.end_track();
    }
    LogInfo.end_track();
    return done && iter == 1;  // Didn't make any updates
  }

  public void run() {
    eventsOut = IOUtils.openOutHard(Execution.getFile("events"));

    initModel();
    generateExamples();
    estimateMoments();
    optimize();
  }

  public static void main(String[] args) {
    Execution.run(args, new Main(), "main", opts);
  }
}
