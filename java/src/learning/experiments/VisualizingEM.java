package learning.experiments;

import fig.basic.*;
import fig.exec.Execution;
import learning.common.Counter;
import learning.models.MixtureOfGaussians;
import learning.models.Params;
import learning.models.loglinear.Example;
import learning.models.loglinear.Models;
import learning.models.loglinear.ParamsVec;
import learning.spectral.applications.ParameterRecovery;
import learning.unsupervised.ExpectationMaximization;
import org.ejml.simple.SimpleMatrix;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.Random;

/**
 * An attempt to visualize EM using CCA
 */
public class VisualizingEM implements Runnable {

  @Option(gloss="Number of parameter guesses to make")
  public static double guesses = 1e4;

  @Option(gloss="Number of EM attempts to make")
  public static double attempts = 1e2;

  @Option(gloss="Number of samples to draw")
  public static double samples = 1e4;

  @Option(gloss="Data used")
  public static double N = 1e4;

  @Option(name = "number of components")
  public static int K = 2;

  @Option(name = "number of dimensions")
  public static int D = 2;

  @Option(name = "number of observables")
  public static int L = 3;

  @Option(name = "Random generator for the initial parameters")
  public static Random trueRandom = new Random(42);

  @Option(name = "Variance in randoms")
  public static double sigma = 1.0;

  @Option(name = "Random generator for the data")
  public static Random genRandom = new Random(12);

  @Option(name = "Random generator for the parameters")
  public static Random initRandom = new Random(23);

  @Override
  public void run() {
    Models.MixtureModel model = new Models.MixtureModel(K,D,L);
    // Generate a bunch of data and report estimates of the likelihood.
    Params trueParams = model.newParams();
    trueParams.initRandom(trueRandom, sigma);

    String outputPath = Execution.getFile("points.dat");
    try( PrintWriter output = new PrintWriter(new FileOutputStream(outputPath))) {
      Counter<Example> data =  (samples < 2e6)
              ? model.drawSamples(trueParams, genRandom, (int) samples)
              : model.getDistribution(trueParams);

      output.write( String.format("%s %f # True params\n", Fmt.D(trueParams.toArray()), model.getLogLikelihood(trueParams, data) -  model.getLogLikelihood(trueParams)));

      Params guessParams = model.newParams();
      ExpectationMaximization em = new ExpectationMaximization();
      // Run EM a bunch of times and record their final point
      for(int attempt = 0; attempt < attempts; attempt++ ) {
        guessParams.initRandom(initRandom, 1.5 * sigma);
        guessParams = em.solveEM(model, data, guessParams);

        output.write(String.format( "%s %f # EM params\n", Fmt.D(guessParams.toArray()), model.getLogLikelihood(guessParams, data) -  model.getLogLikelihood(guessParams) ));
      }

      for(int guess = 0; guess < guesses; guess++) {
        guessParams.initRandom(initRandom, 1.5 * sigma);
        output.write(String.format( "%s %f # Random params\n", Fmt.D(guessParams.toArray()), model.getLogLikelihood(guessParams, data) -  model.getLogLikelihood(guessParams) ));
      }
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    Execution.run(args, new VisualizingEM());
  }
}
