package learning.experiments;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;
import learning.common.Counter;
import learning.models.Params;
import learning.models.loglinear.Example;
import learning.models.loglinear.Model;
import learning.models.loglinear.Models;
import learning.unsupervised.ExpectationMaximization;

import static learning.common.Utils.writeStringHard;

import java.util.Random;

/**
 * Script to run EM
 */
public class RunEM implements  Runnable {
  public static class Options {
    @Option(gloss="Seed for parameters") public Random trueParamsRandom = new Random(44);
    @Option(gloss="Seed for generated data") public Random genRandom = new Random(42);
    @Option(gloss="Noise") public double trueParamsNoise = 1.0;
    @Option(gloss="K") public int K = 2;
    @Option(gloss="D") public int D = 3;
    @Option(gloss="L") public int L = 1;

    @Option(gloss="data points") public int genNumTs = (int) 1e6;
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
    Counter<Example> data =  modelA.drawSamples(trueParams, opts.genRandom, opts.genNumTs);
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

    ExpectationMaximization<Example> em = new ExpectationMaximization<>();
    theta =  em.solveEM(modelA, data, theta);

    Counter<Example> dist = modelA.getDistribution(trueParams);
    Counter<Example> dist_ = modelA.getDistribution(theta);

    for( Example ex : dist ) {
      LogInfo.logs("%s: %f vs %f", ex, dist.getCount(ex), dist_.getCount(ex));
    }
//      List<T> hiddenStates = generateTs(L);
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

}
