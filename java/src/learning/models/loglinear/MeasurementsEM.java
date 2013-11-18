package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import learning.utils.Counter;

import static fig.basic.LogInfo.*;

import java.util.List;

/**
 * Implementation of EM for the measurements Bayesian model in the measurements framework (ref below).
 *
 *   Learning from measurements in exponential families
 *   Percy Liang, Michael I. Jordan, Dan Klein
 *   http://machinelearning.org/archive/icml2009/papers/393.pdf
 */
public class MeasurementsEM implements Runnable {

  @Option(gloss="Regularization for theta") public double thetaRegularization = 1e-5;
  @Option(gloss="Regularization for beta") public double betaRegularization = 1e-5;

  @Option(gloss="Number of iterations optimizing E") public int eIters = 100;
  @Option(gloss="Number of iterations optimizing M") public int mIters = 1;
  @Option(gloss="Number of iterations") public int iters = 100;

  @Option(gloss="Type of optimization to use") public boolean useLBFGS = true;
  @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
  @OptionSet(name="backtrack") public BacktrackingLineSearch.Options backtrack = new BacktrackingLineSearch.Options();

  Maximizer newMaximizer() {
    if (useLBFGS) return new LBFGSMaximizer(backtrack, lbfgs);
    return new GradientMaximizer(backtrack);
  }

  void doGradientCheck(Maximizer.FunctionState state) {
    double epsilon = 1e-4;
    // Save point
    double[] point = state.point();
    double[] gradient = state.gradient();
    double[] currentGradient = gradient.clone();
    double[] currentPoint = point.clone();


    // Set point to be +/- gradient
    for( int i = 0; i < currentPoint.length; i++ ) {
      point[i] = currentPoint[i] + epsilon;
      double valuePlus = state.value();
      point[i] = currentPoint[i] - epsilon;
      double valueMinus = state.value();
      point[i] = currentPoint[i];

      double expectedValue = (valuePlus - valueMinus)/(2*epsilon);
      double actualValue = currentGradient[i];
      assert MatrixOps.equal(expectedValue, actualValue, 1e-4);
    }
  }

  /**
   * Implements the E objective
   *    - L = (tau, \beta) - \sum_i B(\beta ; X_i) - 1/2 betaRegularization \|\beta\|^2
   *    - dL = tau - \sum_i \E_\beta(\sigma(Y_i, X_i)) - 1/betaRegularization \beta
   */
  class MeasurementsEObjective implements Maximizer.FunctionState {
    Model modelA, modelB;
    Counter<Example> X;
    ParamsVec theta, beta, tau;
    ParamsVec gradient;

    double objective, objectiveOffset;
    boolean objectiveValid, gradientValid;
    ParamsVec params, paramsGradient;

    public MeasurementsEObjective(Model modelA, Model modelB, Counter<Example> X, ParamsVec tau, ParamsVec theta, ParamsVec beta) {
      this.modelA = modelA;
      this.modelB = modelB;
      this.X = X;
      this.tau = tau;
      this.theta = theta;
      this.beta = beta;
      this.gradient = new ParamsVec(beta);
      this.params = modelB.newParamsVec();
      this.paramsGradient = modelB.newParamsVec();

      updateOffset();
    }

    public void updateOffset() {
      // Compute the offset to the value
      // $\sum_i A(\theta; X_i) + h_\theta(\theta)
      objectiveOffset = 0;
      // TODO: Should I include the value of A here?
      objectiveOffset += 0.5 * thetaRegularization * theta.dot(theta);
    }

    @Override
    public void invalidate() { objectiveValid = gradientValid = false; }

    @Override
    public double[] point() { return beta.weights; }

    @Override
    /**
     * Compute the value of the E-objective
     * $L = \langle\tau, \beta\rangle + \sum_i A(\theta; X_i) - \sum_i B(\theta, \beta; X_i)
     *          + h_\theta(\theta) - h^*_\beta(\beta)$.
     * The only things that change are B and h^*(\beta).
     */
    public double value() {
      if( objectiveValid ) return (objective + objectiveOffset);
      objective = 0.;

      // Add a linear term for (\tau, \beta)
      objective += tau.dot(beta);

      // Go through each example, and add - B(\theta, X_i)
      params.clear();
      params = ParamsVec.plus(theta, beta, params);
      for( Example X_i : X ) {
         Hypergraph<Example> Hq = modelB.createHypergraph(X_i, params.weights, null, 0);
         Hq.computePosteriors(false);
         double logZ = Hq.getLogZ();
         objective -= X.getCount(X_i) * logZ / X.sum();
      }
      // Finally, subtract regularizer h^*(\beta) = 0.5 \|\beta\|^2
      objective -= 0.5 * betaRegularization * beta.dot(beta);

      return (objective + objectiveOffset);
    }

    @Override
    public double[] gradient() {
      if( gradientValid ) return gradient.weights;

      gradient.clear();
      // Add a term for \tau
      gradient.incr(1.0, tau);

      // Compute expected counts - \E_q(\sigma)
      params.clear();
      params = ParamsVec.plus(theta, beta, params);
      paramsGradient.clear();
      for( Example X_i : X ) {
         Hypergraph<Example> Hq = modelB.createHypergraph(X_i, params.weights, paramsGradient.weights, - X.getCount(X_i)/ X.sum());
         Hq.computePosteriors(false);
         Hq.fetchPosteriors(false);
      }
      gradient = ParamsVec.plus(gradient, paramsGradient, gradient);

      // subtract \nabla h^*(\beta) = \beta
      gradient.incr(-betaRegularization, beta);

      return gradient.weights;
    }
  }

  /**
   * Implements the M objective
   *    - L = \sum_i A(\theta ; X_i) + thetaRegularization/2 \|\theta\|^2
   *    - dL = \sum_i \E_\theta(\phi(Y_i, X_i)) + thetaRegularization \theta
   */
  class MeasurementsMObjective implements Maximizer.FunctionState {
    Model modelA, modelB;
    Counter<Example> X;
    ParamsVec theta, beta, tau;
    ParamsVec params, phi_sigma;
    ParamsVec gradient;

    double objective, objectiveOffset;
    boolean objectiveValid, gradientValid;
    Hypergraph<Example> Hp;

    public MeasurementsMObjective(Model modelA, Model modelB, Counter<Example> X, ParamsVec tau, ParamsVec theta, ParamsVec beta) {
      this.modelA = modelA;
      this.modelB = modelB;
      this.X = X;
      this.tau = tau;
      this.theta = theta;
      this.beta = beta;
      this.gradient = new ParamsVec(theta);
      this.params = modelB.newParamsVec();
      this.phi_sigma = modelB.newParamsVec();

      Hp = modelA.createHypergraph(modelA.L, theta.weights, gradient.weights, 1.);

      updateOffset();
    }

    public void updateOffset() {
      // Compute the offset to the value
      // $(tau, \beta) - h_\beta(\heta)
      params.clear();
      params = ParamsVec.plus(theta, beta, params);

      objectiveOffset = 0;
      // -H(q) = \theta_0^T \E_q[\phi(x,y)] + \beta_0^T \E_q[\phi(x,y)] -
      // \sum_i B
      phi_sigma.clear();
      for( Example X_i : X ) {
        Hypergraph<Example> Hq = modelB.createHypergraph(X_i, params.weights, phi_sigma.weights, X.getCount(X_i)/X.sum());
        Hq.computePosteriors(false);
        Hq.fetchPosteriors(false);
        objectiveOffset -=  X.getCount(X_i) * Hq.getLogZ() / X.sum() ;
      }
      objectiveOffset += theta.dot( phi_sigma );
      objectiveOffset += beta.dot( phi_sigma );

      // h_\sigma( \tau - \E[\sigma(x,y) );
      ParamsVec tau_hat = new ParamsVec(tau);
      ParamsVec.project( phi_sigma, tau_hat );
      objectiveOffset += 0.5  * betaRegularization * MatrixOps.norm( ParamsVec.minus(tau, tau_hat).weights );
    }

    @Override
    public void invalidate() { objectiveValid = gradientValid = false; }

    @Override
    public double[] point() { return theta.weights; }

    @Override
    /**
     * Compute the value of the M-objective
     * $L = \langle\tau, \beta\rangle + \sum_i A(\theta; X_i) - \sum_i B(\theta, \beta; X_i)
     *          + h_\theta(\theta) - h^*_\beta(\beta)$.
     * The only things that change are A, B and h(\theta).
     */
    public double value() {
      if( objectiveValid ) return -(objective + objectiveOffset);

      objective = 0.;

      // Go through each example, and compute A(\theta;X_i)
      Hp.computePosteriors(false);
      objective +=  Hp.getLogZ();
      // - \theta^T \E_q[\phi(X,Y)]
      objective -= theta.dot(phi_sigma);
      // Finally, add regularizer h(\theta) = 0.5 \|\theta\|^2
      objective += 0.5 * thetaRegularization * theta.dot(theta);

      return -(objective + objectiveOffset);
    }

    @Override
    public double[] gradient() {
      if( gradientValid ) return gradient.weights;

      gradient.clear();

      Hp.computePosteriors(false);
      Hp.fetchPosteriors(false);

      ParamsVec phi = modelA.newParamsVec();
      ParamsVec.project( phi_sigma, phi );
      gradient.incr( -1.0, phi );

      // Add \nabla h(\theta)
      gradient.incr(thetaRegularization, theta);

      // Change the sign
      ListUtils.multMut(gradient.weights, -1);

      return gradient.weights;
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
      if( state instanceof MeasurementsEObjective ) {
        ((MeasurementsEObjective)state).updateOffset();
      }
      else if( state instanceof MeasurementsMObjective ) {
        ((MeasurementsMObjective)state).updateOffset();
      }
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
   * @param modelB - exponential family model that has the partition function $B$ above
   * @param data - The $X_i$'s
   * @param measurements - The $\tau$ above
   * @param theta - initial parameters for $\theta$
   * @param beta - initial parameters for $\beta$
   * @return - (theta, beta) that optimize.
   */
  Pair<ParamsVec,ParamsVec> solveMeasurements(
          Model modelA,
          Model modelB,
          Counter<Example> data,
          ParamsVec measurements,
          ParamsVec theta,
          ParamsVec beta
          ) {
    LogInfo.begin_track("solveMeasurements");
    LogInfo.logs( "Solving measurements objective with %d + %d parameters, using %f instances (%d unique)",
            theta.numFeatures, beta.numFeatures, data.sum(), data.size() );

    Maximizer eMaximizer = newMaximizer();
    Maximizer mMaximizer = newMaximizer();

    // Create the E-objective (for $\beta$) - main computations are the partition function for B and expected counts
    MeasurementsEObjective eObjective = new MeasurementsEObjective(modelA, modelB, data, measurements, theta, beta);
    // Create the M-objective (for $\theta$) - main computations are the partition function for A, B and expected counts
    MeasurementsMObjective mObjective = new MeasurementsMObjective(modelA, modelB, data, measurements, theta, beta);
    // Initialize
    eObjective.updateOffset(); mObjective.updateOffset();

    boolean done = false;
    PrintWriter out = IOUtils.openOutHard(Execution.getFile("events"));
    for( int i = 0; i < iters && !done; i++ ) {

      assert eObjective.theta == mObjective.theta;
      assert eObjective.beta == mObjective.beta;

      List<String> items = new ArrayList<>();
      items.add("iter = " + i);
      items.add("eObjective = " + eObjective.value());
      items.add("mObjective = " + mObjective.value());
      LogInfo.logs( StrUtils.join(items, "\t") );
      out.println( StrUtils.join(items, "\t") );
      out.flush();

      // Optimize each one-by-one
      boolean done1 = optimize( eMaximizer, eObjective, "E-" + i, eIters );
      eObjective.updateOffset();
      mObjective.updateOffset();
      eObjective.invalidate();
      mObjective.invalidate();
      LogInfo.logs( "==== midway: eObjective = %f, mObjective = %f", eObjective.value(), mObjective.value() );
      boolean done2 = optimize( mMaximizer, mObjective, "M-" + i, mIters );
      eObjective.updateOffset();
      mObjective.updateOffset();
      eObjective.invalidate();
      mObjective.invalidate();
      LogInfo.logs("==== midway: eObjective = %f, mObjective = %f", eObjective.value(), mObjective.value());

      done = done1 && done2;
    }

    LogInfo.end_track("solveMeasurements");

    return Pair.makePair(theta, beta);
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
      Hypergraph<Example> Hp = model.createHypergraph(L, ex, params, null, 0);
      Hp.computePosteriors(false);
      cnt += data.getCount(ex);
      lhood += data.getCount(ex) * (Hp.getLogZ() - lhood) / cnt;
    }
    return lhood;
  }

  Pair<Model, Model> createModels() {
    LogInfo.begin_track("Creating models");
    // Create two simple models
    Models.MixtureModel modelA = new Models.MixtureModel();
    modelA.K = opts.K;
    modelA.D = opts.D;
    modelA.L = opts.L;
    modelA.createHypergraph(opts.L, null, null, null, 0);

    Models.MixtureModel modelB = new Models.MixtureModel();
    modelB.K = modelA.K;
    modelB.D = modelA.D;
    modelB.L = modelA.L;
    modelB.createHypergraph(opts.L, null, null, null, 0);
    LogInfo.end_track("Creating models");

    return Pair.makePair( (Model) modelA, (Model) modelB );
  }

  public void run(){

    Pair<Model, Model> models = createModels();
    Model modelA = models.getFirst();
    Model modelB = models.getSecond();

    // Create some data
    LogInfo.begin_track("Creating data");
    ParamsVec trueParams = modelA.newParamsVec();
    for(int i = 0; i < trueParams.weights.length; i++)
      trueParams.weights[i] = Math.sin(i);
//      trueParams.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
    trueParams.write(Execution.getFile("true.params"));

    ParamsVec trueMeasurements = modelA.newParamsVec();
    Hypergraph<Example> Hp = modelA.createHypergraph(opts.L, null, trueParams.weights, trueMeasurements.weights, 1.);
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
    logs("Generated %d examples", data.size());
    LogInfo.end_track("Creating data");

    // Fit
    LogInfo.begin_track("Fitting model");

    // Initializing stuff
    ParamsVec theta = new ParamsVec(trueParams);
    ParamsVec beta = modelB.newParamsVec(); //new ParamsVec(trueParams);
    theta.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
    beta.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);

    LogInfo.logs("likelihood(true): " + computeLogZ(modelA, opts.L, trueParams.weights, data) );
    LogInfo.logs("likelihood(est.): " + computeLogZ(modelA, opts.L, theta.weights, data) );

    ParamsVec measurements = modelA.newParamsVec();
    theta.write(Execution.getFile("fit0.params"));
    measurements.clear();
    Hp = modelA.createHypergraph(opts.L, null, theta.weights, measurements.weights, 1);
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);
    measurements.write(Execution.getFile("fit0.counts"));

    // Measurements
    measurements.clear();
    for (Example ex : data) {
      Hypergraph<Example> Hq = modelA.createHypergraph(opts.L, ex, trueParams.weights, measurements.weights, data.getCount(ex)/data.sum());
      Hq.computePosteriors(false);
      Hq.fetchPosteriors(false);
    }
    measurements.write(Execution.getFile("measurements.counts"));

    solveMeasurements( modelA, modelB, data, measurements, theta, beta);

    measurements.clear();
    Hp = modelA.createHypergraph(opts.L, null, theta.weights, measurements.weights, 1);
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
    Execution.putOutput("fit.beta", MatrixFactory.fromVector(beta.weights));
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
