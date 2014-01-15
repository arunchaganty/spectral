package learning.experiments;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionSet;
import fig.exec.Execution;
import learning.models.MixtureOfGaussians;
import learning.spectral.applications.ParameterRecovery;
import org.ejml.simple.SimpleMatrix;

/**
 * Check recovery under model misspecification
 */
public class ModelMisspecification implements Runnable {

  @Option(gloss="K used to guess the model", required=true)
  public static int guessK;

  @Option(gloss="Data used")
  public static double N = 1e4;

  @OptionSet(name = "genOpts")
  public static MixtureOfGaussians.GenerationOptions options = new MixtureOfGaussians.GenerationOptions();

  @Override
  public void run() {
    MixtureOfGaussians model = MixtureOfGaussians.generate(options);
    MixtureOfGaussians model_ = ParameterRecovery.recoverGMM(guessK, (int)N, model, 0.0);
    LogInfo.logs(model.getMeans()[0]);
    LogInfo.logs(model_.getMeans()[0]);

    // Generate a bunch of data and report the likelihood.
    SimpleMatrix[] data = model.sample((int)N);
    LogInfo.logs("True likelihood %f", model.computeLikelihood(data)/N);
    LogInfo.logs("Fit likelihood %f", model_.computeLikelihood(data)/N);
  }

  public static void main(String[] args) {
    Execution.run(args, new ModelMisspecification());
  }
}
