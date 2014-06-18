package learning.experiments;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;
import learning.common.Counter;
import learning.models.HiddenMarkovModel;
import learning.models.loglinear.Example;
import org.ejml.simple.SimpleMatrix;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Random;

import static fig.basic.LogInfo.begin_track;
import static fig.basic.LogInfo.end_track;
import static learning.common.Utils.outputList;
import static learning.common.Utils.outputListF;

/**
 * Optimize the piecewise likelihood or do linear algebra. Which would you put a million dollars on?
 */
public class PiecewiseVsAlgebra implements  Runnable {
  @Option(gloss="Data used")
  public double N = 1e2;

  @Option( gloss = "number of states" )
  public int K = 2;
  @Option( gloss = "number of symbols" )
  public int D = 2;
  @Option(gloss="Sequence length")
  public int L = 3;

  @Option( gloss = "Generator for parameters" )
  public Random paramsRandom = new Random(1);
  @Option( gloss = "variance for parameters" )
  public double paramsNoise = 1.0;

  @Option( gloss = "Generator for data" )
  public Random genRandom = new Random(1);

  @Option(gloss="Generator for initial points")
  public Random initRandom = new Random(42);
  @Option(gloss="Variance")
  public double initRandomNoise = 1.0;

  @Option(gloss="Start at the exact solution?")
  public boolean initExact = false;
  @Option(gloss="Start at the exact solution?")
  public boolean initExactO = false;

  @Option(gloss="iterations to run EM")
  public int iters = 100;

  @Option(gloss="Run EM?")
  public boolean runEM = false;

  @Option(gloss="How much to smooth")
  public double smoothMeasurements = 1e-2;


  // TODO: Generalize to use only params
  double getPiecewiseLikelihood(HiddenMarkovModel model, HiddenMarkovModel.Parameters trueParams, HiddenMarkovModel.Parameters params, Counter<Example> data) {
    int K = model.getK();

    double lhood = 0.;
    // Run EM to recover pi
    double[][] O = trueParams.getO();
    double[] pi = params.getPi();

    for(Example ex : data){
      double weight = data.getFraction(ex);

      int x1 = ex.x[0];
      double z = 0.;
      for(int h = 0; h < K; h++) z += pi[h] * O[h][x1];
      lhood += weight * Math.log( z );
    }

    return lhood;
  }

  public void run() {
    HiddenMarkovModel model = new HiddenMarkovModel(K, D, 3);
    // Initialize model
    begin_track("Generating model");
    HiddenMarkovModel.Parameters trueParams = model.newParams();
    trueParams.initRandom(paramsRandom, paramsNoise);

    // Get data
    Counter<Example> data;
    if(N >= 1e7) // Seriously too much data
      data =  model.getDistribution(trueParams);
    else
      data =  model.drawSamples(trueParams, genRandom, (int) N);

    end_track("Generating model");

    HiddenMarkovModel.Parameters params = trueParams.newParams();

    // - Ha
    assert K == 2;

    SimpleMatrix M1 = new SimpleMatrix(1,D);
    {
      for(Example ex : data) {
        double weight = data.getFraction(ex);

        int x1 = ex.x[0];
        M1.set(x1, M1.get(x1) + weight);
      }
    }

    SimpleMatrix O = new SimpleMatrix(trueParams.getO());
    SimpleMatrix pi = new SimpleMatrix(1, K);

    double lhood0 = getPiecewiseLikelihood(model, trueParams, trueParams, data);


    OutputStreamWriter out;
    try{
      out = new OutputStreamWriter(new FileOutputStream(Execution.getFile("out.tab")));
    } catch(IOException ex) {
      out = new OutputStreamWriter(System.out);
    }

    for(double pi0 = 0.; pi0 <= 1.0; pi0 += 0.01) {
      params.set(model.piFeature(0), pi0);
      params.set(model.piFeature(1), 1. - pi0);
      double lhood = getPiecewiseLikelihood(model, trueParams, params, data);

      pi.set(0, pi0);
      pi.set(1, 1. - pi0);

      double obj = Math.pow(pi.mult(O).minus(M1).normF(),2);
      LogInfo.log(outputListF(out,
              "pi0", pi0,
              "lhood", -(lhood - lhood0),
              "obj", obj
      ));
    }
    try {
      out.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    Execution.run(args, new PiecewiseVsAlgebra());
  }
}
