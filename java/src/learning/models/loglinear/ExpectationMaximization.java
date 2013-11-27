package learning.models.loglinear;

import fig.basic.*;
import fig.exec.Execution;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import learning.utils.Counter;

import java.io.PrintWriter;
import java.util.*;

import static learning.utils.UtilsJ.doGradientCheck;

/**
 * Expectation Maximization for a model
 */
public class ExpectationMaximization implements Runnable {

  @Option(gloss="Regularization for theta") public double thetaRegularization = 0; //1e-3;

  @Option(gloss="Number of iterations") public int iters = 400;
  @Option(gloss="Number of iterations") public int mIters = 1000;

  @Option(gloss="Type of optimization to use") public boolean useLBFGS = true;
  @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
  @OptionSet(name="backtrack") public BacktrackingLineSearch.Options backtrack = new BacktrackingLineSearch.Options();

  Maximizer newMaximizer() {
    if (useLBFGS) return new LBFGSMaximizer(backtrack, lbfgs);
    return new GradientMaximizer(backtrack);
  }

  /**
   * Implements the M objective
   *    - q = \E_{\theta}(z | x)
   *    - dL = tau - \sum_i \E_\beta(\sigma(Y_i, X_i)) - 1/betaRegularization \beta
   */
  class Objective implements Maximizer.FunctionState {
    Model modelA;
    Counter<Example> X;
    ParamsVec theta;
    ParamsVec gradient;
    final ParamsVec q; // Marginal distribution $z|x$
//    final Hypergraph<Example> Hp;

    double objective;
    boolean objectiveValid, gradientValid;

    public Objective(Model modelA, Counter<Example> X, ParamsVec theta, final ParamsVec q) {
      this.modelA = modelA;
      this.X = X;
      this.theta = theta;
      this.gradient = new ParamsVec(theta);

      this.q = q;
//      Hp = modelA.createHypergraph(theta.weights, gradient.weights, -1.);
    }

    @Override
    public void invalidate() { objectiveValid = gradientValid = false; }

    @Override
    public double[] point() { return theta.weights; }

    @Override
    /**
     * Compute the value of the E-objective
     * $L(\theta; q) = 1/|X| \sum_x \sum_z q(z | x) * \theta^T \phi(x,z) - \log Z(\theta) - \frac{1}{2}\eta_\theta|\theta^2|^2$
     * $dL(\theta; q)/d\theta = 1/|X| \sum_x \sum_z q(z | x) * \phi(x,z) - E_{\theta}[\phi(x,z)] - \eta_\theta \theta $
     */
  public double value() {
      if( objectiveValid ) return (objective);
      objective = 0.;


      // Go through each example, and add 1/|X| \sum_x \sum_z q(z | x) * \theta^T \phi(x,z)
      objective += theta.dot(q);

      // Subtract the log partition function - log Z(\theta)
      Hypergraph<Example> Hp = modelA.createHypergraph(theta.weights, null, 0.);
      Hp.computePosteriors(false);
      objective -= Hp.getLogZ();

      // Finally, subtract regularizer = 0.5 \eta_\theta \|\theta\|^2
//      objective += 0.5 * thetaRegularization * theta.dot(theta);

      return objective;
    }

    @Override
    public double[] gradient() {
      if( gradientValid ) return gradient.weights;

      gradient.clear();


      // Go through each example, and add 1/|X| \sum_x \sum_z q(z | x) * \phi(x,z)
      gradient.incr(1.0, q);

      // Subtract the log partition function - log Z(\theta)
      Hypergraph<Example> Hp = modelA.createHypergraph(theta.weights, gradient.weights, -1.);
      Hp.computePosteriors(false);
      Hp.fetchPosteriors(false);

      // subtract \nabla h^*(\beta) = \beta
//      gradient.incr(-thetaRegularization, theta);

//      for( int i = 0; i < gradient.weights.length; i++ ) gradient.weights[i] *= -1.;

      return gradient.weights;
    }
  }

  public void updateMarginals(Model modelA, ParamsVec params, Counter<Example> X, ParamsVec q) {
    q.clear();
    for( Example X_i : X ) {
      Hypergraph<Example> Hq = modelA.createHypergraph(X_i, params.weights, q.weights, X.getCount(X_i)/X.sum());
      Hq.computePosteriors(false);
      Hq.fetchPosteriors(false);
    }
  }

  boolean optimize( Maximizer maximizer, Maximizer.FunctionState state, String label, int numIters ) {
    LogInfo.begin_track("optimize " + label);
    state.invalidate();
    boolean done = false;
    // E-step
    int iter;

    PrintWriter out = IOUtils.openOutHard(Execution.getFile(label + ".events"));

    double oldObjective = Double.NEGATIVE_INFINITY;

    for (iter = 0; iter < numIters && !done; iter++) {
      state.invalidate();

      // Logging stuff
      List<String> items = new ArrayList<>();
      items.add("iter = " + iter);
      items.add("objective = " + state.value());
      items.add("pointNorm = " + MatrixOps.norm(state.point()));
      items.add("gradientNorm = " + MatrixOps.norm(state.gradient()));
      LogInfo.logs( StrUtils.join(items, "\t") );
      out.println( StrUtils.join(items, "\t") );
      out.flush();

      double objective = state.value();
      assert objective > oldObjective;
      oldObjective = objective;

      done = maximizer.takeStep(state);
    }
    // Do a gradient check only at the very end.
    doGradientCheck(state);

    List<String> items = new ArrayList<>();
    items.add("iter = " + iter);
    items.add("objective = " + state.value());
    items.add("pointNorm = " + MatrixOps.norm(state.point()));
    items.add("gradientNorm = " + MatrixOps.norm(state.gradient()));
    LogInfo.logs( StrUtils.join(items, "\t") );
    out.println( StrUtils.join(items, "\t") );
    out.flush();

    LogInfo.end_track();
    return done && iter == 1;  // Didn't make any updates
  }

  /**
   * Find parameters that optimize the measurements objective:
   * $L = \langle\tau, \beta\rangle + \sum_i A(\theta; X_i) - \sum_i B(\theta, \beta; X_i)
   *          + h_\theta(\theta) - h^*_\beta(\beta)$.
   * @param modelA - exponential family model that has the partition function $A$ above
   * @param data - The $X_i$'s
   * @param theta - initial parameters for $\theta$
   * @return - (theta, beta) that optimize.
   */
  ParamsVec solveEM(
          Model modelA,
          Counter<Example> data,
          ParamsVec theta
          ) {
    LogInfo.begin_track("solveEM");
    LogInfo.logs( "Solving EM objective with %d parameters, using %f instances (%d unique)",
            theta.numFeatures, data.sum(), data.size() );

    Maximizer mMaximizer = newMaximizer();

    final ParamsVec marginals = modelA.newParamsVec();

    // Create the M-objective (for $\theta$) - main computations are the partition function for A, B and expected counts
    Objective mObjective = new Objective(modelA, data, theta, marginals);
    updateMarginals(modelA, theta, data, marginals);

    boolean done = false;
    PrintWriter out = IOUtils.openOutHard(Execution.getFile("events"));
    for( int i = 0; i < iters && !done; i++ ) {
      List<String> items = new ArrayList<>();
      items.add("iter = " + i);
      items.add("mObjective = " + mObjective.value());
      LogInfo.logs( StrUtils.join(items, "\t") );
      out.println( StrUtils.join(items, "\t") );
      out.flush();

      // Optimize each one-by-one
      // Get marginals
      updateMarginals(modelA, theta, data, marginals);
      done = optimize( mMaximizer, mObjective, "M-" + i, mIters );
      mObjective.invalidate();
    }

    LogInfo.end_track("solveEM");

    return theta;
  }

  public static class Options {
    @Option(gloss="Seed for parameters") public Random trueParamsRandom = new Random(42);
    @Option(gloss="Seed for generated data") public Random genRandom = new Random(42);
    @Option(gloss="Noise") public double trueParamsNoise = 0.01;
    @Option(gloss="K") public int K = 2;
    @Option(gloss="D") public int D = 2;
    @Option(gloss="L") public int L = 1;

    @Option(gloss="data points") public int genNumExamples = 100;
  }

  public static Options opts = new Options();

  public double computeLogZ(Model model, int L, double[] params, Counter<Example> data) {
    double lhood = 0.0;
    int cnt = 0;
    for( Example ex: data) {
      Hypergraph<Example> Hp = model.createHypergraph(ex, params, null, 0);
      Hp.computePosteriors(false);
      cnt += data.getCount(ex);
      lhood += data.getCount(ex) * (Hp.getLogZ() - lhood) / cnt;
    }
    return lhood;
  }

  Model createModels() {
    LogInfo.begin_track("Creating models");
    // Create two simple models
    Models.MixtureModel modelA = new Models.MixtureModel(opts.K, opts.D, opts.L);
    modelA.createHypergraph(null, null, null, 0);

    LogInfo.end_track("Creating models");

    return modelA;
  }

  public void run(){

    Model modelA = createModels();

    // Create some data
    LogInfo.begin_track("Creating data");
    ParamsVec trueParams = modelA.newParamsVec();
    for(int i = 0; i < trueParams.weights.length; i++)
      trueParams.weights[i] = Math.sin(i);
//      trueParams.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
    trueParams.write(Execution.getFile("true.params"));

    ParamsVec trueMeasurements = modelA.newParamsVec();
    Hypergraph<Example> Hp = modelA.createHypergraph(null, trueParams.weights, trueMeasurements.weights, 1.);
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);
    trueMeasurements.write(Execution.getFile("true.counts"));

    // Generate examples from the model
    Counter<Example> data = new Counter<>();
    for (int i = 0; i < opts.genNumExamples; i++) {
      Example ex = modelA.newExample();
      Hp.fetchSampleHyperpath(opts.genRandom, ex);
      data.add(ex);
    }
    LogInfo.logs("Generated %d examples", data.size());
    LogInfo.end_track("Creating data");

    // Fit
    LogInfo.begin_track("Fitting model");

    // Initializing stuff
    ParamsVec theta = new ParamsVec(trueParams);
    theta.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);

    LogInfo.logs("likelihood(true): " + computeLogZ(modelA, opts.L, trueParams.weights, data) );
    LogInfo.logs("likelihood(est.): " + computeLogZ(modelA, opts.L, theta.weights, data) );

    ParamsVec measurements = modelA.newParamsVec();
    theta.write(Execution.getFile("fit0.params"));
    measurements.clear();
    Hp = modelA.createHypergraph(null, theta.weights, measurements.weights, 1);
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);
    measurements.write(Execution.getFile("fit0.counts"));

    // Measurements
    measurements.clear();
    for (Example ex : data) {
      Hypergraph<Example> Hq = modelA.createHypergraph(ex, trueParams.weights, measurements.weights, data.getCount(ex)/data.sum());
      Hq.computePosteriors(false);
      Hq.fetchPosteriors(false);
    }
    measurements.write(Execution.getFile("true.counts"));

    solveEM(modelA, data, theta);

    measurements.clear();
    Hp = modelA.createHypergraph(null, theta.weights, measurements.weights, 1);
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);

    int[] perm = new int[trueMeasurements.K];

    double error = theta.computeDiff(trueParams, perm);
    Execution.putOutput("params-error", error);
    LogInfo.logs("params error: " + error + " " + Fmt.D(perm));

    error = measurements.computeDiff(trueMeasurements, perm);
    Execution.putOutput("counts-error", error);
    LogInfo.logs("counts error: " + error + " " + Fmt.D(perm));
    LogInfo.logs("likelihood(true): " + computeLogZ(modelA, opts.L, trueParams.weights, data) );
    LogInfo.logs("likelihood(est.): " + computeLogZ(modelA, opts.L, theta.weights, data) );

    // Compute the likelihoods
    theta.write(Execution.getFile("fit.params"));
    measurements.write(Execution.getFile("fit.counts"));

    LogInfo.end_track("Fitting model");
  }

  /**
   * Run the measurements objective on some trivially simple problem
   * @param args
   */
  public static void main(String[] args) {
    Execution.run(args, new MeasurementsEM(), "main", opts);
  }
}

