package learning.unsupervised;

import fig.basic.*;
import fig.exec.Execution;
import learning.common.Counter;
import learning.models.ExponentialFamilyModel;
import learning.models.Params;

import static learning.common.Utils.optimize;
import static learning.common.Utils.outputList;

/**
 * Expectation Maximization for directed model
 */
public class DirectedExpectationMaximization<T> {

  @Option(gloss="Number of iterations")
  public int iters = 1000;
  @Option(gloss="Convergence")
  public double eps = 1e-4;
  @Option(gloss="print some diagnostic data")
  public boolean diagnosticMode = false;

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

    double oldObjective = modelA.getLogLikelihood(theta, data);

    LogInfo.log(outputList(
            "iter", 0,
            "likelihood", oldObjective
    ));

    Params marginals = theta.newParams();
    boolean done = false;
    for( int i = 0; i < iters && !done; i++ ) {
      marginals.clear();
      modelA.updateMarginals(theta, data, 1.0, marginals);
      theta.copyOver(marginals);
      double objective = modelA.getLogLikelihood(theta, data);

      LogInfo.log(outputList(
              "iter", i+1,
              "likelihood", objective
      ));

      assert objective > oldObjective;
      done = (objective - oldObjective < eps);
      oldObjective = objective;
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

