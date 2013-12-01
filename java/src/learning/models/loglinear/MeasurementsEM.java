package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import learning.models.ExponentialFamilyModel;
import learning.utils.Counter;
import learning.utils.UtilsJ;

import static fig.basic.LogInfo.*;
import static learning.utils.UtilsJ.doGradientCheck;

import java.util.List;

/**
 * Implementation of EM for the measurements Bayesian model in the measurements framework (ref below).
 *
 *   Learning from measurements in exponential families
 *   Percy Liang, Michael I. Jordan, Dan Klein
 *   http://machinelearning.org/archive/icml2009/papers/393.pdf
 */
public class MeasurementsEM implements Runnable {
  @Option(gloss="Regularization for theta") public double thetaRegularization = 1e-3;
  @Option(gloss="Regularization for beta") public double betaRegularization = 1e-3;

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

  /**
   * Implements the E objective
   *    - L = (tau, \beta) - \sum_i B(\beta ; X_i) - 1/2 betaRegularization \|\beta\|^2
   *    - dL = tau - \sum_i \E_\beta(\sigma(Y_i, X_i)) - 1/betaRegularization \beta
   */
  class MeasurementsEObjective implements Maximizer.FunctionState {
    ExponentialFamilyModel<Example> modelA;
    ExponentialFamilyModel<Example> modelB;
    Counter<Example> X;
    final ParamsVec theta, beta, tau;
    final ParamsVec gradient;

    double objective, objectiveOffset;
    boolean objectiveValid, gradientValid;
    final ParamsVec params, paramsGradient;

    public MeasurementsEObjective(ExponentialFamilyModel<Example> modelA, ExponentialFamilyModel<Example> modelB, Counter<Example> X, ParamsVec tau, ParamsVec theta, ParamsVec beta) {
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
      objectiveOffset += 0.5 * thetaRegularization * theta.dot(theta);

      objectiveOffset += modelA.getLogLikelihood(theta);
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
      if( objectiveValid ) return objective;
      objective = 0.;

      // Add a linear term for (\tau, \beta)
      objective += tau.dot(beta);

      // Go through each example, and add - B(\theta, X_i)
      params.clear();
      ParamsVec.plus(theta, beta, params);
      objective -= modelB.getLogLikelihood(params, X);
      // Finally, subtract regularizer h^*(\beta) = 0.5 \|\beta\|^2
      objective -= 0.5 * betaRegularization * beta.dot(beta);

      return objective;
//      return (objective + objectiveOffset);
    }

    @Override
    public double[] gradient() {
      if( gradientValid ) return gradient.weights;

      gradient.clear();
      // Add a term for \tau
      gradient.incr(1.0, tau);

      // Compute expected counts - \E_q(\sigma)
      params.clear();
      ParamsVec.plus(theta, beta, params);
      paramsGradient.copy( modelB.getMarginals(params, X) );
      ParamsVec.minus(gradient, paramsGradient, gradient);

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
    ExponentialFamilyModel<Example> modelA;
    ExponentialFamilyModel<Example> modelB;
    Counter<Example> X;
    final ParamsVec theta, beta, tau;
    final ParamsVec params, phi_sigma;
    final ParamsVec gradient;

    double objective, objectiveOffset;
    boolean objectiveValid, gradientValid;

    public MeasurementsMObjective(ExponentialFamilyModel<Example> modelA, ExponentialFamilyModel<Example> modelB, Counter<Example> X, ParamsVec tau, ParamsVec theta, ParamsVec beta) {
      this.modelA = modelA;
      this.modelB = modelB;
      this.X = X;
      this.tau = tau;
      this.theta = theta;
      this.beta = beta;
      this.gradient = new ParamsVec(theta);
      this.params = modelB.newParamsVec();
      this.phi_sigma = modelB.newParamsVec();

      updateOffset();
    }

    public void updateOffset() {
      // Compute the offset to the value
      // $(tau, \beta) - h_\beta(\heta)
      params.clear();
      ParamsVec.plus(theta, beta, params);

      objectiveOffset = 0;
      // -H(q) = \theta_0^T \E_q[\phi(x,y)] + \beta_0^T \E_q[\phi(x,y)] -
      // \sum_i B
      phi_sigma.clear();
      phi_sigma.copy( modelB.getMarginals(params, X) );

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
      if( objectiveValid ) return objective;

      objective = 0.;
      // Go through each example, and compute A(\theta;X_i)
      objective += modelA.getLogLikelihood(theta);
      // - \theta^T \E_q[\phi(X,Y)]
      objective -= theta.dot(phi_sigma);
      // Finally, add regularizer h(\theta) = 0.5 \|\theta\|^2
      objective += 0.5 * thetaRegularization * theta.dot(theta);

//      objective = -(objective + objectiveOffset);
      objective *= -1.;

      return objective;
    }

    @Override
    public double[] gradient() {
      if( gradientValid ) return gradient.weights;

      gradient.copy(modelA.getMarginals(theta));

//      ParamsVec phi = modelA.newParamsVec();
//      ParamsVec.project( phi_sigma, phi );
      gradient.incr( -1.0, phi_sigma );

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
   *
   * @param modelA - exponential family model that has the partition function $A$ above
   * @param modelB - exponential family model that has the partition function $B$ above
   * @param data - The $X_i$'s
   * @param measurements - The $\tau$ above
   * @param theta - initial parameters for $\theta$
   * @param beta - initial parameters for $\beta$
   * @return - (theta, beta) that optimize.
   */
  Pair<ParamsVec,ParamsVec> solveMeasurements(
          ExponentialFamilyModel<Example> modelA,
          ExponentialFamilyModel<Example> modelB,
          Counter<Example> data,
          ParamsVec measurements,
          ParamsVec theta,
          ParamsVec beta
          ) {
    LogInfo.begin_track("solveMeasurements");
    LogInfo.logs( "Solving measurements objective with %d + %d parameters, using %f instances (%d unique)",
            theta.numFeatures, beta.numFeatures, data.sum(), data.size() );

    Execution.putOutput( "measuredFraction", measurements.numFeatures / (float) theta.numFeatures );

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
    ParamsVec oldTheta = new ParamsVec(theta);
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
      mObjective.updateOffset();
      mObjective.invalidate();
      LogInfo.logs( "==== midway: eObjective = %f, mObjective = %f", eObjective.value(), mObjective.value() );
      boolean done2 = optimize( mMaximizer, mObjective, "M-" + i, mIters );
      eObjective.updateOffset();
      eObjective.invalidate();
      LogInfo.logs("==== midway: eObjective = %f, mObjective = %f", eObjective.value(), mObjective.value());

      System.arraycopy(theta.weights, 0, oldTheta.weights, 0, theta.weights.length);

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

  public double computeLogZ(ExponentialFamilyModel<Example> model, ParamsVec params, Counter<Example> data) {
    return model.getLogLikelihood(params, data);
  }

  Pair<ExponentialFamilyModel<Example>, ExponentialFamilyModel<Example>> createModels() {
    LogInfo.begin_track("Creating models");
    // Create two simple models
    Models.MixtureModel modelA = new Models.MixtureModel();
    modelA.K = opts.K;
    modelA.D = opts.D;
    modelA.L = opts.L;

    Models.MixtureModel modelB = new Models.MixtureModel();
    modelB.K = modelA.K;
    modelB.D = modelA.D;
    modelB.L = modelA.L;
    LogInfo.end_track("Creating models");

    return Pair.makePair( (ExponentialFamilyModel<Example>)modelA, (ExponentialFamilyModel<Example>)modelB );
  }

  public void run(){

    Pair<ExponentialFamilyModel<Example>, ExponentialFamilyModel<Example>> models = createModels();
    ExponentialFamilyModel<Example> modelA = models.getFirst();
    ExponentialFamilyModel<Example> modelB = models.getSecond();

    // Create some data
    LogInfo.begin_track("Creating data");
    ParamsVec trueParams = modelA.newParamsVec();
    for(int i = 0; i < trueParams.weights.length; i++)
      trueParams.weights[i] = Math.sin(i);
//      trueParams.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
    trueParams.write(Execution.getFile("true.params"));

    ParamsVec trueMeasurements = modelA.getMarginals(trueParams);
    trueMeasurements.write(Execution.getFile("true.counts"));

    // Generate examples from the model
    Counter<Example> data = modelA.drawSamples(trueParams, opts.genRandom, opts.genNumExamples);
    logs("Generated %d examples", data.size());
    LogInfo.end_track("Creating data");

    // Fit
    LogInfo.begin_track("Fitting model");

    // Initializing stuff
    ParamsVec theta = new ParamsVec(trueParams);
    theta.write(Execution.getFile("fit0.params"));

    ParamsVec beta = modelB.newParamsVec(); //new ParamsVec(trueParams);
    theta.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
    beta.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);

    LogInfo.logs("likelihood(true): " + computeLogZ(modelA, trueParams, data) );
    LogInfo.logs("likelihood(est.): " + computeLogZ(modelA, theta, data) );


    ParamsVec measurements = modelA.getMarginals(theta);
    measurements.write(Execution.getFile("fit0.counts"));

    // Measurements
    measurements = modelA.getMarginals(trueParams, data);
    measurements.write(Execution.getFile("measurements.counts"));

    solveMeasurements( modelA, modelB, data, measurements, theta, beta);

    measurements = modelA.getMarginals(theta);

    int[] perm = new int[trueMeasurements.K];

    double error = theta.computeDiff(trueParams, perm);
    Execution.putOutput("params-error", error);
    LogInfo.logs("params error: " + error + " " + Fmt.D(perm));

    error = measurements.computeDiff(trueMeasurements, perm);
    Execution.putOutput("counts-error", error);
    LogInfo.logs("counts error: " + error + " " + Fmt.D(perm));
    LogInfo.logs("likelihood(true): " + computeLogZ(modelA, trueParams, data) );
    LogInfo.logs("likelihood(est.): " + computeLogZ(modelA, theta, data) );

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
