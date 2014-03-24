package learning.unsupervised;

import fig.basic.*;
import fig.exec.Execution;
import learning.models.ExponentialFamilyModel;
import learning.models.Params;
import learning.models.loglinear.Example;
import learning.models.loglinear.Model;
import learning.models.loglinear.Models;
import learning.common.Counter;

import java.io.PrintWriter;
import java.util.*;

import static learning.common.Utils.optimize;
import static learning.common.Utils.outputList;
import static learning.common.Utils.writeStringHard;

/**
 * Expectation Maximization for a model
 */
public class ExpectationMaximization implements Runnable {

  @Option(gloss="Regularization for theta") public double thetaRegularization = 1e-5; //0; //1e-3;

  @Option(gloss="Number of iterations") public int iters = 1000;
  @Option(gloss="Number of iterations") public int mIters = 1;

  @Option(gloss="Diagnostic mode") public boolean diagnosticMode = false;

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
  public class Objective implements Maximizer.FunctionState {
    final ExponentialFamilyModel<Example> modelA;
    final Counter<Example> X;
    final Params theta;
    final Params gradient;
    final Params q; // Marginal distribution $z|x$
    final int[] histogram;

    double objective;
    boolean objectiveValid, gradientValid;

    public Objective(ExponentialFamilyModel<Example> modelA, Counter<Example> X, Params theta, final Params q) {
      this.modelA = modelA;
      this.X = X;
      this.theta = theta;
      this.gradient = theta.copy();

      histogram = computeHistogram(X);

      this.q = q;
    }

    @Override
    public void invalidate() { objectiveValid = gradientValid = false; theta.invalidateCache(); }

    @Override
    public double[] point() { return theta.toArray(); }

    @Override
    /**
     * Compute the value of the E-objective
     * $L(\theta; q) = 1/|X| \sum_x \sum_z q(z | x) * \theta^T \phi(x,z) - \log Z(\theta) - \frac{1}{2}\eta_\theta|\theta^2|^2$
     * $dL(\theta; q)/d\theta = 1/|X| \sum_x \sum_z q(z | x) * \phi(x,z) - E_{\theta}[\phi(x,z)] - \eta_\theta \theta $
     */
  public double value() {
      if( objectiveValid ) return (objective);
      // Every time the point changes, re-cache
      theta.cache();

      objective = 0.;
      // Go through each example, and add 1/|X| \sum_x \sum_z q(z | x) * \theta^T \phi(x,z)
      objective += theta.dot(q);

      // Subtract the log partition function - log Z(\theta)
      objective -= modelA.getLogLikelihood(theta, histogram);

      // Finally, subtract regularizer = 0.5 \eta_\theta \|\theta\|^2
      objective -= 0.5 * thetaRegularization * theta.dot(theta);

      objectiveValid = true;
      return objective;
    }

    @Override
    public double[] gradient() {
      if( gradientValid ) return gradient.toArray();
      // Every time the point changes, re-cache
      theta.cache();

      gradient.clear();

      // Go through each example, and add 1/|X| \sum_x \sum_z q(z | x) * \phi(x,z)
      gradient.plusEquals(1.0, q);

      // Subtract the log partition function - log Z(\theta)
      modelA.updateMarginals(theta, histogram, -1.0, gradient);
//      Params marginals = modelA.getMarginals(theta, histogram);
//      gradient.plusEquals(-1.0, marginals);

      // subtract \nabla h^*(\beta) = \beta
      gradient.plusEquals(-thetaRegularization, theta);

      gradientValid = true;
      return gradient.toArray();
    }
  }

  static int[] computeHistogram(Counter<Example> examples) {
    // Get max length
    int maxLength = 0;
    for(Example ex : examples) {
      if(ex.x.length > maxLength) maxLength = ex.x.length;
    }
    // Populate
    int[] histogram = new int[maxLength+1];
    for(Example ex : examples) {
      histogram[ex.x.length]++;
    }

    return histogram;
  }

  public class EMState {
    public ExponentialFamilyModel<Example> model;
    public Counter<Example> data;
    public int[] histogram;

    final Maximizer maximizer;
    public final Params marginals;
    public final Params theta;

    public final Objective objective;

    public EMState(
            ExponentialFamilyModel<Example> model,
            Counter<Example> data,
            Params theta
    ) {
      LogInfo.logs( "Solving EM objective with %d parameters, using %f instances (%d unique)",
              theta.size(), data.sum(), data.size() );

      this.model = model;
      this.data = data;
      this.theta = theta;
      histogram = computeHistogram(data);

      maximizer = newMaximizer();
      marginals = model.newParams();

      // Create the M-objective (for $\theta$) - main computations are the partition function for A, B and expected counts
      this.objective = new Objective(model, data, theta, marginals);
    }
  }

  public boolean takeStep(EMState state) {
    LogInfo.begin_track("E-step");
    state.objective.invalidate();
    state.theta.cache();

    // Optimize each one-by-one
    // Get marginals
    state.marginals.clear();
    state.model.updateMarginals(state.theta, state.data, 1.0, state.marginals);
    LogInfo.end_track("E-step");

    LogInfo.begin_track("M-step");
    boolean done = optimize(state.maximizer, state.objective, "M", mIters, diagnosticMode);
    state.objective.invalidate();
    LogInfo.end_track("M-step");
    return done;
  }


  public EMState newState(
          ExponentialFamilyModel<Example> modelA,
          Counter<Example> data,
          Params theta
  ) {
    return new EMState(modelA, data, theta);
  }

  /**
   * Find parameters that optimize the measurements objective:
   * $L = \langle\tau, \beta\rangle + \sum_i A(\theta; X_i) - \sum_i B(\theta, \beta; X_i)
   *          + h_\theta(\theta) - h^*_\beta(\beta)$.
   *
   * @param modelA - exponential family model that has the partition function $A$ above
   * @param data - The $X_i$'s
   * @param theta - initial parameters for $\theta$
   * @return - (theta, beta) that optimize.
   */
  public Params solveEM(
          ExponentialFamilyModel<Example> modelA,
          Counter<Example> data,
          Params theta
          ) {
    LogInfo.begin_track("solveEM");
    LogInfo.logs( "Solving EM objective with %d parameters, using %f instances (%d unique)",
            theta.size(), data.sum(), data.size() );
    EMState state = new EMState(modelA, data, theta);

    boolean done = false;
    for( int i = 0; i < iters && !done; i++ ) {
      LogInfo.log(outputList(
              "iter", i,
              "mObjective", state.objective.value(),
              "likelihood", (modelA.getLogLikelihood(state.theta, data) - modelA.getLogLikelihood(state.theta, state.histogram))
      ));
      done = takeStep(state);
    }
    if(done) LogInfo.log("Reached optimum");
    Execution.putOutput("optimization-done", done);

    // Used to diagnose
    if(diagnosticMode) {
      LogInfo.log("Expected: " + modelA.getMarginals(theta));
      LogInfo.log("Data-Expected: " + modelA.getMarginals(theta, data));
      LogInfo.log("Data-Only: " + modelA.getSampleMarginals(data));
    }

    LogInfo.end_track("solveEM");

    return theta;
  }

  public static class Options {
    @Option(gloss="Seed for parameters") public Random trueParamsRandom = new Random(44);
    @Option(gloss="Seed for generated data") public Random genRandom = new Random(42);
    @Option(gloss="Noise") public double trueParamsNoise = 1.0;
    @Option(gloss="K") public int K = 2;
    @Option(gloss="D") public int D = 3;
    @Option(gloss="L") public int L = 1;

    @Option(gloss="data points") public int genNumExamples = (int) 1e6;
  }

  public static Options opts = new Options();

  Model createModels() {
    LogInfo.begin_track("Creating models");
    // Create two simple models
    Models.MixtureModel modelA = new Models.MixtureModel(opts.K, opts.D, opts.L);
    LogInfo.end_track("Creating models");

    return modelA;
  }

  public void run(){
    Model modelA = createModels();

    // Create some data
    LogInfo.begin_track("Creating data");
    Params trueParams = modelA.newParams();
    {
      double[] params = trueParams.toArray();
      for(int i = 0; i < params.length; i++)
        params[i] = Math.sin(i);
//      params[i] = Math.sin(i);
    }
//    trueParams.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
    trueParams.write(Execution.getFile("true.params"));

    Params trueMeasurements = modelA.getMarginals(trueParams);
    trueMeasurements.write(Execution.getFile("true.counts"));

    // Generate examples from the model
    Counter<Example> data =  modelA.drawSamples(trueParams, opts.genRandom, opts.genNumExamples);
    LogInfo.logs("Generated %d examples", data.size());
    LogInfo.end_track("Creating data");

    // Fit
    LogInfo.begin_track("Fitting model");

    // Initializing stuff
    Params theta = trueParams.copy();
    theta.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
    writeStringHard(Execution.getFile("fit0.params"), theta.toString());

    LogInfo.log("likelihood(true): " + modelA.getLogLikelihood(trueParams, data));
    LogInfo.log("likelihood(est.): " +  modelA.getLogLikelihood(theta, data));

    Params measurements;

    measurements = modelA.getMarginals(theta);
    writeStringHard(Execution.getFile("fit0.counts"), measurements.toString());

    theta = solveEM(modelA, data, theta);

    Counter<Example> dist = modelA.getDistribution(trueParams);
    Counter<Example> dist_ = modelA.getDistribution(theta);

    for( Example ex : dist ) {
      LogInfo.logs("%s: %f vs %f", ex, dist.getCount(ex), dist_.getCount(ex));
    }
//      List<Example> hiddenStates = generateExamples(L);
//    measurements = modelA.getMarginals(theta);
//    int[] perm = new int[trueMeasurements.K];
//
//    double error = theta.computeDiff(trueParams, perm);
//    Execution.putOutput("params-error", error);
//    LogInfo.logs("params error: " + error + " " + Fmt.D(perm));
//
//    error = measurements.computeDiff(trueMeasurements, perm);
//    Execution.putOutput("counts-error", error);
//    LogInfo.logs("counts error: " + error + " " + Fmt.D(perm));
    LogInfo.log("likelihood(true): " + modelA.getLogLikelihood(trueParams, data));
    LogInfo.log("likelihood(est.): " + modelA.getLogLikelihood(theta, data));

    // Compute the likelihoods
    theta.write(Execution.getFile("fit.params"));
////    measurements.write(Execution.getFile("fit.counts"));
////
    LogInfo.end_track("Fitting model");
  }

  /**
   * Run the measurements objective on some trivially simple problem
   * @param args
   */
  public static void main(String[] args) {
    Execution.run(args, new ExpectationMaximization(), "main", opts);
  }
}

