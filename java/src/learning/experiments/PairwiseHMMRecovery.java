package learning.experiments;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionSet;
import fig.exec.Execution;
import learning.models.HiddenMarkovModel;
import learning.models.MixtureOfGaussians;
import learning.spectral.applications.ParameterRecovery;
import org.ejml.simple.SimpleMatrix;

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

  @Option(gloss="Guess K")
  public int guessK = 2;

  @OptionSet(name = "genOpts") public HiddenMarkovModel.GenerationOptions options = new HiddenMarkovModel.GenerationOptions();

  public double computeLikelihood(HiddenMarkovModel model, int[][] data) {
    double lhood = 0.;
    for(int i = 0; i < data.length; i++) {
      lhood += (model.likelihood(data[i]) - lhood)/(++i);
    }

    return lhood;
  }

  public int[][] sample(HiddenMarkovModel model, int N) {
    int[][] samples = new int[N][L];
    for(int i = 0; i < N; i++)
      samples[i] = model.sample(L);
    return samples;
  }

  @Override
  public void run() {
    HiddenMarkovModel model = HiddenMarkovModel.generate(options);
    HiddenMarkovModel model_ = ParameterRecovery.recoverHMM(guessK, (int) N, model, 0.0);
    LogInfo.log(model.getO());
    LogInfo.log(model_.getO());

    // Generate a bunch of data and report the likelihood.
    int[][] data = sample(model, (int) N);
    LogInfo.logs("True likelihood %f", computeLikelihood(model, data));
    LogInfo.logs("Fit likelihood %f", computeLikelihood(model_, data));
  }

  public static void main(String[] args) {
    Execution.run(args, new PairwiseHMMRecovery());
  }
}
