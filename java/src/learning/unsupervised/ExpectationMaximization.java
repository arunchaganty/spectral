package learning.unsupervised;

import fig.basic.*;
import fig.exec.Execution;
import learning.models.ExponentialFamilyModel;
import learning.models.Params;
import learning.common.Counter;

import static learning.common.Utils.optimize;
import static learning.common.Utils.outputList;

/**
 * Expectation Maximization for a model
 */
public class ExpectationMaximization<T> {

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
    final ExponentialFamilyModel<T> modelA;
    final Counter<T> X;
    final Params theta;
    final Params gradient;
    final Params q; // Marginal distribution $z|x$
    final int[] histogram;

    double objective;
    boolean objectiveValid, gradientValid;

    public Objective(ExponentialFamilyModel<T> modelA, Counter<T> X, Params theta, final Params q) {
      this.modelA = modelA;
      this.X = X;
      this.theta = theta;
      this.gradient = theta.copy();

      histogram = computeHistogram(modelA, X);

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

  static <T> int[] computeHistogram(ExponentialFamilyModel<T> model, Counter<T> examples) {
    // Get max length
    int maxLength = 0;
    for(T ex : examples) {
      if(model.getSize(ex) > maxLength) maxLength = model.getSize(ex);
    }
    // Populate
    int[] histogram = new int[maxLength+1];
    for(T ex : examples) {
      histogram[model.getSize(ex)]++;
    }

    return histogram;
  }

  public class EMState {
    public ExponentialFamilyModel<T> model;
    public Counter<T> data;
    public int[] histogram;

    final Maximizer maximizer;
    public final Params marginals;
    public final Params theta;

    public final Objective objective;

    public EMState(
            ExponentialFamilyModel<T> model,
            Counter<T> data,
            Params theta
    ) {
      LogInfo.logs( "Solving EM objective with %d parameters, using %f instances (%d unique)",
              theta.size(), data.sum(), data.size() );

      this.model = model;
      this.data = data;
      this.theta = theta;
      histogram = computeHistogram(model, data);

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
          ExponentialFamilyModel<T> modelA,
          Counter<T> data,
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
          ExponentialFamilyModel<T> modelA,
          Counter<T> data,
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
}

