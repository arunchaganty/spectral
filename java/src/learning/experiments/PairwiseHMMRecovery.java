package learning.experiments;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionSet;
import fig.exec.Execution;
import learning.models.BasicParams;
import learning.models.HiddenMarkovModel;
import learning.spectral.applications.ParameterRecovery;

import java.util.Random;

import static learning.common.Utils.outputList;

/**
 * Recover a HMM from pairwise factors
 * - First solve for observations
 * - Then solve the convex likelihood.
 */
public class PairwiseHMMRecovery implements  Runnable {
  @Option(gloss="Data used")
  public double N = 1e5;
  @Option(gloss="Sequence length")
  public int L = 5;

  @OptionSet(name = "genOpts")
  public HiddenMarkovModel.GenerationOptions options = new HiddenMarkovModel.GenerationOptions();

  @Option(gloss="generation")
  public Random genRand = new Random(42);

  @Option(gloss="init")
  public Random initRandom = new Random(42);
  @Option(gloss="init")
  public double initRandomNoise = 1.0;

  @Option(gloss="iterations to run EM")
  public int iters = 100;

  @Option(gloss="Run EM?")
  public boolean runEM = true;

  @Option(gloss="How much to smooth")
  public double smoothMeasurements = 1e-2;

  enum RunMode {
    EM,
    EMGoodStart,
    SpectralInitialization,
    SpectralConvex
  }
  @Option
  public RunMode mode = RunMode.EM;

  @Override
  public void run() {
    int D = options.emissionCount;
    int K = options.stateCount;

    HiddenMarkovModel model = HiddenMarkovModel.generate(options);
    // Get data
    int[][] X = model.sample(genRand, (int)N, L);

    BasicParams params = model.toParams();

    Execution.putOutput("true-likelihood", model.likelihood(X));
    Execution.putOutput("true-paramsError", model.toParams().computeDiff(params, null));
    Execution.putOutput("true-pi", model.getPi());
    Execution.putOutput("true-T", model.getT());
    Execution.putOutput("true-O", model.getO());

    LogInfo.log(outputList(
            "true-likelihood", model.likelihood(X),
            "true-paramsError", model.toParams().computeDiff(params, null)
    ));

    // Process via EM or Spectral
    HiddenMarkovModel model_;
    switch(mode) {
      case EM: {
        model_ = new HiddenMarkovModel(
                HiddenMarkovModel.Params.uniformWithNoise(initRandom, K, D, initRandomNoise));
        runEM = true; // Regardless of what you said before.
      } break;
      case EMGoodStart: {
        model_ = new HiddenMarkovModel( model.getParams().clone() );
        runEM = true; // Regardless of what you said before.
      } break;
      case SpectralInitialization: {
        model_ = ParameterRecovery.recoverHMM(K, (int)N, model, smoothMeasurements);
      } break;
      default:
        throw new RuntimeException("Not implemented");
    }

    Execution.putOutput("initial-likelihood", model_.likelihood(X));
    Execution.putOutput("initial-paramsError", model_.toParams().computeDiff(params, null));
    Execution.putOutput("initial-pi", model_.getPi());
    Execution.putOutput("initial-T", model_.getT());
    Execution.putOutput("initial-O", model_.getO());

    if(runEM) {
      double lhood_ = Double.NEGATIVE_INFINITY;
      for(int i = 0; i < iters; i++) {
        // Print error per iteration.
        double lhood = model_.likelihood(X);
        LogInfo.log(outputList(
                "iter", i,
                "likelihood", lhood,
                "paramsError", params.computeDiff(model_.toParams(), null)
        ));

        assert lhood > lhood_;
        lhood_ = lhood;
        model_.baumWelchStep(X);
      }
      LogInfo.log(outputList(
              "likelihood", model_.likelihood(X),
              "paramsError", params.computeDiff(model_.toParams(), null)
      ));
    }

    Execution.putOutput("final-likelihood", model_.likelihood(X));
    Execution.putOutput("final-paramsError", params.computeDiff(model_.toParams(), null));
    Execution.putOutput("final-pi", model_.getPi());
    Execution.putOutput("final-T", model_.getT());
    Execution.putOutput("final-O", model_.getO());
  }

  public static void main(String[] args) {
    Execution.run(args, new PairwiseHMMRecovery());
  }
}
