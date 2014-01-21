package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import learning.linalg.MatrixOps;
import learning.models.ExponentialFamilyModel;
import learning.models.Params;
import learning.utils.Counter;

import static fig.basic.LogInfo.*;
import static learning.utils.UtilsJ.optimize;
import static learning.utils.UtilsJ.writeStringHard;

import java.util.List;

/**
 * Implementation of EM for the measurements Bayesian model in the measurements framework (ref below).
 *
 *   Learning from measurements in exponential families
 *   Percy Liang, Michael I. Jordan, Dan Klein
 *   http://machinelearning.org/archive/icml2009/papers/393.pdf
 */
public class MeasurementsEM implements Runnable {
  @Option(gloss="Regularization for theta") public double thetaRegularization = 0;// 1e-3;
  @Option(gloss="Regularization for beta") public double betaRegularization = 0; //1e-3;

  @Option(gloss="Number of iterations optimizing E") public int eIters = 100;
  @Option(gloss="Number of iterations optimizing M") public int mIters = 1;
  @Option(gloss="Number of iterations") public int iters = 100;

  @Option(gloss="Type of optimization to use") public boolean useLBFGS = true;
  @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
  @OptionSet(name="backtrack") public BacktrackingLineSearch.Options backtrack = new BacktrackingLineSearch.Options();

  @Option(gloss="Do extra careful checks") public boolean diagnosticMode = false; //true;

  Maximizer newMaximizer() {
    backtrack.verbose = 5;
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
    final Params theta, beta, tau;
    final Params gradient;

    double objective, objectiveOffset;
    boolean objectiveValid, gradientValid;
    final Params params, paramsGradient;

    public MeasurementsEObjective(ExponentialFamilyModel<Example> modelA, ExponentialFamilyModel<Example> modelB, Counter<Example> X, Params tau, Params theta, Params beta) {
      this.modelA = modelA;
      this.modelB = modelB;
      this.X = X;
      this.tau = tau;
      this.theta = theta;
      this.beta = beta;
      this.gradient = beta.copy();
      this.params = modelB.newParams();
      this.paramsGradient = modelB.newParams();

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
    public double[] point() { return beta.toArray(); }

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
      params.plusEquals(1.0, beta);
      params.plusEquals(1.0, theta);
      objective -= modelB.getLogLikelihood(params, X);
      // Finally, subtract regularizer h^*(\beta) = 0.5 \|\beta\|^2
      objective -= 0.5 * betaRegularization * beta.dot(beta);

      return objective;
//      return (objective + objectiveOffset);
    }

    @Override
    public double[] gradient() {
      if( gradientValid ) return gradient.toArray();

      gradient.clear();
      // Add a term for \tau
      gradient.plusEquals(1.0, tau);

      // Compute expected counts - \E_q(\sigma)
      params.clear();
      params.plusEquals(1.0, theta);
      params.plusEquals(1.0, beta);
      paramsGradient.copyOver(modelB.getMarginals(params, X));
      gradient.plusEquals(-1.0, paramsGradient);

      // subtract \nabla h^*(\beta) = \beta
      gradient.plusEquals(-betaRegularization, beta);

      return gradient.toArray();
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
    final Params theta, beta, tau;
    final Params params, phi_sigma;
    final Params gradient;

    double objective, objectiveOffset;
    boolean objectiveValid, gradientValid;

    public MeasurementsMObjective(ExponentialFamilyModel<Example> modelA, ExponentialFamilyModel<Example> modelB, Counter<Example> X, Params tau, Params theta, Params beta) {
      this.modelA = modelA;
      this.modelB = modelB;
      this.X = X;

      this.tau = tau;
      this.theta = theta;
      this.beta = beta;

      this.gradient = theta.newParams();

      this.params = modelB.newParams();
      this.phi_sigma = modelB.newParams();

      updateOffset();
    }

    public void updateOffset() {
      // Compute the offset to the value
      // $(tau, \beta) - h_\beta(\heta)
      params.clear();
      params.plusEquals(1.0, theta);
      params.plusEquals(1.0, beta);

//      objectiveOffset = 0;
      // -H(q) = \theta_0^T \E_q[\phi(x,y)] + \beta_0^T \E_q[\phi(x,y)] -
      // \sum_i B
      phi_sigma.clear();
      phi_sigma.copyOver(modelB.getMarginals(params, X));

      // h_\sigma( \tau - \E[\sigma(x,y) );
//      Params tau_hat = tau.copy();
//      tau_hat.copyOver(phi_sigma);
//      objectiveOffset += 0.5  * betaRegularization * MatrixOps.norm( tau.plus(-1.0, tau_hat).toArray() );
    }

    @Override
    public void invalidate() { objectiveValid = gradientValid = false; }

    @Override
    public double[] point() { return theta.toArray(); }

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
      if( gradientValid ) return gradient.toArray();

      Params marginals = modelA.getMarginals(theta);
      gradient.copyOver(marginals);

//      Params phi = modelA.newParams();
//      Params.project( phi_sigma, phi );
      gradient.plusEquals(-1.0, phi_sigma);

      // Add \nabla h(\theta)
      gradient.plusEquals(thetaRegularization, theta);

      // Change the sign
      gradient.scaleEquals(-1.0);

      return gradient.toArray();
    }
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
  Pair<Params,Params> solveMeasurements(
          ExponentialFamilyModel<Example> modelA,
          ExponentialFamilyModel<Example> modelB,
          Counter<Example> data,
          Params measurements,
          Params theta,
          Params beta
          ) {
    LogInfo.begin_track("solveMeasurements");
    LogInfo.logs( "Solving measurements objective with %d + %d parameters, using %f instances (%d unique)",
            theta.size(), beta.size(), data.sum(), data.size() );

    if(Execution.getActualExecDir() != null)
      Execution.putOutput( "actualMeasuredFraction", measurements.size() / (float) theta.size() );

    Maximizer eMaximizer = newMaximizer();
    Maximizer mMaximizer = newMaximizer();

    // Create the E-objective (for $\beta$) - main computations are the partition function for B and expected counts
    MeasurementsEObjective eObjective = new MeasurementsEObjective(modelA, modelB, data, measurements, theta, beta);
    // Create the M-objective (for $\theta$) - main computations are the partition function for A, B and expected counts
    MeasurementsMObjective mObjective = new MeasurementsMObjective(modelA, modelB, data, measurements, theta, beta);
    // Initialize
    if(diagnosticMode) {
      eObjective.updateOffset(); mObjective.updateOffset();
    }
    boolean done = false;
    PrintWriter out = null;
    if(Execution.getActualExecDir() != null)
      out = IOUtils.openOutHard(Execution.getFile("events"));
    Params oldTheta = theta.copy();
    for( int i = 0; i < iters && !done; i++ ) {

      assert eObjective.theta == mObjective.theta;
      assert eObjective.beta == mObjective.beta;

      List<String> items = new ArrayList<>();
      items.add("iter = " + i);
      items.add("eObjective = " + eObjective.value());
      items.add("mObjective = " + mObjective.value());
      LogInfo.log(StrUtils.join(items, "\t"));
      if(out != null){
        out.println( StrUtils.join(items, "\t") );
        out.flush();
      }

      // Optimize each one-by-one
      boolean done1 = optimize( eMaximizer, eObjective, "E-" + i, eIters, diagnosticMode );
      mObjective.updateOffset();
      mObjective.invalidate();
      if(diagnosticMode) {
      }
      LogInfo.logs("==== midway: eObjective = %f, mObjective = %f", eObjective.value(), mObjective.value());
      boolean done2 = optimize( mMaximizer, mObjective, "M-" + i, mIters, diagnosticMode );
      eObjective.updateOffset();
      eObjective.invalidate();
      if(diagnosticMode) {
      }
      LogInfo.logs("==== midway: eObjective = %f, mObjective = %f", eObjective.value(), mObjective.value());

      oldTheta.copyOver(theta);

      done = done1 && done2;
    }

    LogInfo.end_track("solveMeasurements");

    return Pair.newPair(theta, beta);
  }

  public static class Options {
    @Option(gloss="Seed for parameters") public Random trueParamsRandom = new Random(42);
    @Option(gloss="Seed for generated data") public Random genRandom = new Random(42);
    @Option(gloss="Noise") public double trueParamsNoise = 0.01;
    @Option(gloss="K") public int K = 2;
    @Option(gloss="D") public int D = 2;
    @Option(gloss="L") public int L = 3;

    @Option(gloss="data points") public int genNumExamples = 100;
  }

  public static Options opts = new Options();

  public double computeLogZ(ExponentialFamilyModel<Example> model, Params params, Counter<Example> data) {
    return model.getLogLikelihood(params, data);
  }

  Pair<ExponentialFamilyModel<Example>, ExponentialFamilyModel<Example>> createModels() {
    LogInfo.begin_track("Creating models");
    // Create two simple models
    Models.MixtureModel modelA = new Models.MixtureModel(opts.K, opts.D, opts.L);

    Models.MixtureModel modelB = new Models.MixtureModel(opts.K, opts.D, opts.L);
    LogInfo.end_track("Creating models");

    return Pair.newPair((ExponentialFamilyModel<Example>) modelA, (ExponentialFamilyModel<Example>) modelB);
  }

  public void run(){

    Pair<ExponentialFamilyModel<Example>, ExponentialFamilyModel<Example>> models = createModels();
    ExponentialFamilyModel<Example> modelA = models.getFirst();
    ExponentialFamilyModel<Example> modelB = models.getSecond();

    // Create some data
    LogInfo.begin_track("Creating data");
    Params trueParams = modelA.newParams();
    for(int i = 0; i < trueParams.size(); i++)
      trueParams.toArray()[i] = Math.sin(i);
//      trueParams.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
    if(Execution.getActualExecDir() != null)
    writeStringHard(Execution.getFile("true.params"), trueParams.toString());
//    trueParams.write(Execution.getFile("true.params"));

    Params trueMeasurements = modelA.getMarginals(trueParams);
    if(Execution.getActualExecDir() != null)
    writeStringHard(Execution.getFile("true.counts"), trueMeasurements.toString());

    // Generate examples from the model
//    Counter<Example> data = modelA.drawSamples(trueParams, opts.genRandom, opts.genNumExamples);
    Counter<Example> data = modelA.getDistribution(trueParams);
    logs("Generated %d examples", data.size());
    LogInfo.end_track("Creating data");

    // Fit
    LogInfo.begin_track("Fitting model");

    // Initializing stuff
    Params theta = trueParams.copy();
    if(Execution.getActualExecDir() != null)
    writeStringHard(Execution.getFile("fit0.params"), theta.toString());

    Params beta = modelB.newParams();
//    beta.copyOver(trueParams);
    Params noise = modelA.newParams();
    noise.initRandom(opts.trueParamsRandom, 1.);
    theta.plusEquals(noise);
//    theta.initRandom(opts.trueParamsRandom, 3.0);
//    theta.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
//    beta.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);

    LogInfo.log("likelihood(true): " + computeLogZ(modelA, trueParams, data));
    LogInfo.log("likelihood(est.): " + computeLogZ(modelA, theta, data));


    // Measurements
    Params measurements = modelA.getMarginals(theta);
    if(Execution.getActualExecDir() != null)
    writeStringHard(Execution.getFile("fit0.counts"), measurements.toString());

    measurements = modelA.getMarginals(trueParams, data);
    if(Execution.getActualExecDir() != null)
    writeStringHard(Execution.getFile("measurements.counts"), measurements.toString());

    solveMeasurements( modelA, modelB, data, measurements, theta, beta);

    measurements = modelA.getMarginals(theta);
    log( Fmt.D(trueMeasurements.toArray()));
    log( Fmt.D(measurements.toArray()));

//    int[] perm = new int[trueMeasurements.K];
//    double error = theta.computeDiff(trueParams, perm);
//    Execution.putOutput("params-error", error);
//    LogInfo.logs("params error: " + error + " " + Fmt.D(perm));
//
//    error = measurements.computeDiff(trueMeasurements, perm);
//    Execution.putOutput("counts-error", error);
//    LogInfo.logs("counts error: " + error + " " + Fmt.D(perm));
    LogInfo.log("likelihood(true): " + computeLogZ(modelA, trueParams, data));
    LogInfo.log("likelihood(est.): " + computeLogZ(modelA, theta, data));

    // Compute the likelihoods
    if(Execution.getActualExecDir() != null)
    writeStringHard(Execution.getFile("fit.params"), theta.toString());
    if(Execution.getActualExecDir() != null)
    Execution.putOutput("fit.beta", Fmt.D(beta.toArray()));
    if(Execution.getActualExecDir() != null)
    writeStringHard(Execution.getFile("fit.counts"), measurements.toString());

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
