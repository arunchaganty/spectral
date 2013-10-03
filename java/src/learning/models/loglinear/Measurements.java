package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;

import static fig.basic.LogInfo.*;

import java.util.List;

/**
 * Implementation of the Measurements framework:
 *   Learning from measurements in exponential families
 *   Percy Liang, Michael I. Jordan, Dan Klein
 *   http://machinelearning.org/archive/icml2009/papers/393.pdf
 */
public class Measurements implements Runnable {

  @Option(gloss="Number of iterations optimizing E") public int eIters = 100;
  @Option(gloss="Number of iterations optimizing M") public int mIters = 100;
  @Option(gloss="Number of iterations") public int iters = 10;

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

  static class MeasurementsEObjective implements Maximizer.FunctionState {
    Model modelA, modelB;
    List<Example> X;
    ParamsVec theta, beta, tau;
    ParamsVec gradient;

    double objective, objectiveOffset;
    boolean objectiveValid, gradientValid;
    ParamsVec params, paramsGradient;

    public MeasurementsEObjective(Model modelA, Model modelB, List<Example> X, ParamsVec tau, ParamsVec theta, ParamsVec beta) {
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
      for( Example X_i : X ) {
        Hypergraph<Example> Hp = modelA.createHypergraph(X_i, theta.weights, null, 0);
        Hp.computePosteriors(false);
        objectiveOffset += Hp.getLogZ();
      }
      objectiveOffset += 0.5 * theta.dot(theta);
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
//      if( objectiveValid ) return objective + objectiveOffset;
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
         objective -= logZ;
      }
      // Finally, subtract regularizer h^*(\beta) = 0.5 \|\beta\|^2
      objective -= 0.5 * beta.dot(beta);

      return objective + objectiveOffset;
    }

    @Override
    public double[] gradient() {
//      if( gradientValid ) return gradient.weights;

      gradient.clear();
      // Add a term for \tau
      gradient.incr(1.0, tau);

      // Compute expected counts - \E_q(\sigma)
      params.clear();
      params = ParamsVec.plus(theta, beta, params);
      paramsGradient.clear();
      for( Example X_i : X ) {
         Hypergraph<Example> Hq = modelB.createHypergraph(X_i, params.weights, paramsGradient.weights, -1);
         Hq.computePosteriors(false);
         Hq.fetchPosteriors(false);
      }
      gradient = ParamsVec.plus(gradient, paramsGradient, gradient);

      // subtract \nabla h^*(\beta) = \beta
      gradient.incr(-1.0, beta);

      return gradient.weights;
    }

    // TODO: Add gradient check
  }

  static class MeasurementsMObjective implements Maximizer.FunctionState {
    Model modelA, modelB;
    List<Example> X;
    ParamsVec theta, beta, tau;
    ParamsVec gradient;

    double objective, objectiveOffset;
    boolean objectiveValid, gradientValid;
    ParamsVec params, paramsGradient;

    public MeasurementsMObjective(Model modelA, Model modelB, List<Example> X, ParamsVec tau, ParamsVec theta, ParamsVec beta) {
      this.modelA = modelA;
      this.modelB = modelB;
      this.X = X;
      this.tau = tau;
      this.theta = theta;
      this.beta = beta;
      this.gradient = new ParamsVec(theta);
      this.params = modelB.newParamsVec();
      this.paramsGradient = modelB.newParamsVec();
    }

    public void updateOffset() {
      // Compute the offset to the value
      // $(tau, \beta) - h_\beta(\heta)

      objectiveOffset = 0;
      objectiveOffset += tau.dot(beta);
      objectiveOffset -= 0.5 * beta.dot(beta);
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
      //if( objectiveValid ) return - (objective + objectiveOffset);

      objective = 0.;

      // Go through each example, and compute A(\theta;X_i) - B(\theta, X_i)
      for( Example X_i : X ) {
        Hypergraph<Example> Hp = modelA.createHypergraph(X_i, theta.weights, null, 0);
        Hp.computePosteriors(false);
        objective +=  Hp.getLogZ();
      }

      params.clear();
      params = ParamsVec.plus(theta, beta, params);
      for( Example X_i : X ) {
        Hypergraph<Example> Hq = modelB.createHypergraph(X_i, params.weights, null, 0);
        Hq.computePosteriors(false);
        objective -=  Hq.getLogZ();
      }
      // Finally, add regularizer h(\theta) = 0.5 \|\theta\|^2
      objective += 0.5 * theta.dot(theta);

      return - (objective + objectiveOffset);
    }

    @Override
    public double[] gradient() {
      //if( gradientValid ) return gradient.weights;

      gradient.clear();

      // Compute expected counts \E_p(\phi) - \E_q(\phi)
      for( Example X_i : X ) {
        Hypergraph<Example> Hp = modelA.createHypergraph(X_i, theta.weights, gradient.weights, 1);
        Hp.computePosteriors(false);
        Hp.fetchPosteriors(false);
      }

      params.clear();
      params = ParamsVec.plus(theta, beta, params);
      paramsGradient.clear();
      for( Example X_i : X ) {
        Hypergraph<Example> Hq = modelB.createHypergraph(X_i, params.weights, paramsGradient.weights, -1);
        Hq.computePosteriors(false);
        Hq.fetchPosteriors(false);
      }
      gradient = ParamsVec.plus(gradient, paramsGradient, gradient);

      // Add \nabla h(\theta)
      gradient.incr(1.0, theta);

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

    for (iter = 0; iter < numIters && !done; iter++) {
      doGradientCheck(state);

      // Logging stuff
      List<String> items = new ArrayList<String>();
      items.add("iter = " + iter);
      items.add("objective = " + state.value());
      items.add("point = " + Fmt.D(state.point()));
      items.add("gradient = " + Fmt.D(state.gradient()));
      LogInfo.logs( StrUtils.join(items, "\t") );
      out.println( StrUtils.join(items, "\t") );
      out.flush();

      done = maximizer.takeStep(state);
    }
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
          List<Example> data,
          ParamsVec measurements,
          ParamsVec theta,
          ParamsVec beta
          ) {
    LogInfo.begin_track("solveMeasurements");
    LogInfo.logs( "Solving measurements objective with %d + %d parameters, using %d instances",
            theta.numFeatures, beta.numFeatures, data.size() );

    Maximizer eMaximizer = newMaximizer();
    Maximizer mMaximizer = newMaximizer();

    // Create the E-objective (for $\beta$) - main computations are the partition function for B and expected counts
    // Create the M-objective (for $\theta$) - main computations are the partition function for A, B and expected counts
    MeasurementsEObjective eObjective = new MeasurementsEObjective(modelA, modelB, data, measurements, theta, beta);
    MeasurementsMObjective mObjective = new MeasurementsMObjective(modelA, modelB, data, measurements, theta, beta);

    boolean done = false;
    PrintWriter out = IOUtils.openOutHard(Execution.getFile("events"));
    for( int i = 0; i < iters && !done; i++ ) {

      assert eObjective.theta == mObjective.theta;
      assert eObjective.beta == mObjective.beta;

      List<String> items = new ArrayList<String>();
      items.add("iter = " + i);
      items.add("eObjective = " + eObjective.value());
      items.add("mObjective = " + mObjective.value());
      LogInfo.logs( StrUtils.join(items, "\t") );
      out.println( StrUtils.join(items, "\t") );
      out.flush();

      // Optimize each one-by-one
      boolean done1 = optimize( eMaximizer, eObjective, "E-" + i, eIters );
      mObjective.updateOffset();
      boolean done2 = optimize( mMaximizer, mObjective, "M-" + i, mIters );
      eObjective.updateOffset();

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

  public void run(){
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

    // Create some data
    LogInfo.begin_track("Creating data");
    ParamsVec trueParams = modelA.newParamsVec();
    trueParams.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
    ParamsVec measurements = modelA.newParamsVec();

    Hypergraph<Example> Hp = modelA.createHypergraph(opts.L, null, trueParams.weights, measurements.weights, 1);
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);
    // Scale measurements by number of examples
    ListUtils.multMut(measurements.weights, opts.genNumExamples);

    trueParams.write(Execution.getFile("true.params"));
    measurements.write(Execution.getFile("true.counts"));

    Execution.putOutput("params.map", trueParams.featureIndexer);
    Execution.putOutput("true.params", MatrixFactory.fromVector(trueParams.weights));
    Execution.putOutput("true.counts", MatrixFactory.fromVector(measurements.weights));

    // Generate examples from the model
    List<Example> data = new ArrayList<Example>();
    for (int i = 0; i < opts.genNumExamples; i++) {
      Example ex = modelA.newExample();
      Hp.fetchSampleHyperpath(opts.genRandom, ex);
      data.add(ex);
    }
    logs("Generated %d examples", data.size());
    LogInfo.end_track("Creating data");

    // Fit
    LogInfo.begin_track("Fitting model");
    ParamsVec theta = new ParamsVec(trueParams);
    ParamsVec beta = new ParamsVec(trueParams);
    theta.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
    beta.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);

    solveMeasurements( modelA, modelB, data, measurements, theta, beta);

    measurements.clear();
    Hp = modelA.createHypergraph(opts.L, null, theta.weights, measurements.weights, 1);
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);
    // Scale measurements by number of examples
    ListUtils.multMut(measurements.weights, opts.genNumExamples);

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
    Execution.run(args, new Measurements(), "main", opts);
  }
}
