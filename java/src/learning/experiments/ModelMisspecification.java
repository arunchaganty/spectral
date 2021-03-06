package learning.experiments;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionSet;
import fig.exec.Execution;
import learning.common.Counter;
import learning.common.Utils;
import learning.models.MixtureOfGaussians;
import learning.models.MultiViewGaussian;
import learning.models.Params;
import learning.spectral.applications.ParameterRecovery;
import learning.unsupervised.DirectedExpectationMaximization;
import learning.unsupervised.ExpectationMaximization;
import learning.unsupervised.ThreeViewMethod;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

/**
 * Check recovery under model misspecification
 */
public class ModelMisspecification implements Runnable {

  @Option(gloss="K used to guess the model", required=true)
  public static int guessK;

  @Option(gloss="Data used")
  public static double N = 1e4;

  @Option(gloss="number of components")
  public static int K = 2;
  @Option(gloss="number of dimensions")
  public static int D = 2;
  @Option(gloss="number of views")
  public static int L = 3;

  @Option(gloss = "random-generator for parameters")
  public static Random trueRandom = new Random(2);
  @Option(gloss = "scale of randomness for parameters")
  public static double trueRandomScale = 1.0;
  @Option(gloss = "random-generator for initializations")
  public static Random initRandom = new Random(3);
  @Option(gloss = "scale of randomness for initializations")
  public static double initRandomScale = 1.0;
  @Option(gloss = "random-generator for generation")
  public static Random genRandom = new Random(4);
  @Option(gloss = "scale of randomness for generation")
  public static double genRandomScale = 1.0;

  @Option(gloss = "Initialize at the true parameters")
  public static boolean initializeAtTrue = true;

  @Option(gloss = "amount to smooth measurements by")
  public double smoothMeasurements;

  @Override
  public void run() {
    // Create model
    MultiViewGaussian model = new MultiViewGaussian(K,D,L);
    Params trueParams = model.newParams();
    trueParams.initRandom(trueRandom, trueRandomScale);

    Counter<double[][]> data = model.drawSamples(trueParams, genRandom, (int)N);
    double true_lhood = -model.getLogLikelihood(trueParams, data);

    LogInfo.log(
            Utils.outputList(
                    "true-params", trueParams,
                    "true-lhood", true_lhood
            ));

    // Initialize

    Params initParams = trueParams.copy();
    if(!initializeAtTrue)
      initParams.initRandom(initRandom, initRandomScale);
    double init_lhood = -model.getLogLikelihood(initParams, data);

    LogInfo.log(
            Utils.outputList(
                    "init-params", initParams,
                    "init-lhood", model.getLogLikelihood(initParams, data),
                    "params-error-init", trueParams.computeDiff(initParams),
                    "lhood-error-init", init_lhood - true_lhood
            ));

    // Run EM
    DirectedExpectationMaximization<double[][]> em = new DirectedExpectationMaximization<>();
    Params emParams = em.solveEM(model, data, initParams);
    for(int h = 0; h < K; h++) emParams.toArray()[model.sigma(h)] = trueParams.toArray()[model.sigma(h)];
    double em_lhood = -model.getLogLikelihood(emParams, data);
    LogInfo.log(
            Utils.outputList(
                    "em-params", emParams,
                    "em-lhood", em_lhood,
                    "params-error-em", trueParams.computeDiff(emParams),
                    "lhood-error-em", em_lhood - true_lhood
            ));

    // Run Spectral
    ThreeViewMethod<double[][]> spectral = new ThreeViewMethod<>();
    Params momParams = spectral.solve(model, data, trueParams, smoothMeasurements);
    for(int h = 0; h < K; h++) momParams.toArray()[model.sigma(h)] = trueParams.toArray()[model.sigma(h)];
    double mom_lhood = -model.getLogLikelihood(momParams, data);
    LogInfo.log(
            Utils.outputList(
                    "mom-params", momParams,
                    "mom-lhood", mom_lhood,
                    "params-error-mom", trueParams.computeDiff(momParams),
                    "lhood-error-mom", mom_lhood - true_lhood
            ));
  }

  public static void main(String[] args) {
    Execution.run(args, new ModelMisspecification());
  }
}
