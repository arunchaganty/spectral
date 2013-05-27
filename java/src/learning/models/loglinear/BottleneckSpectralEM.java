package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;
import learning.linalg.*;
import learning.spectral.TensorMethod;

/**
 * NIPS 2013
 * Uses method of moments to initialize parameters 
 * for EM for general log-linear models.
 *  - Uses Hypergraph framework to represent a model.
 */
public class BottleneckSpectralEM implements Runnable {
  @Option(gloss="Number of optimization outside iterations") public int numIters = 100;
  @Option(gloss="Number of optimization inside E-step iterations") public int eNumIters = 1;
  @Option(gloss="Number of optimization inside M-step iterations") public int mNumIters = 1;
  @Option(gloss="Whether to keep hypergraphs for all the examples") public boolean storeHypergraphs = true;
  @Option(gloss="Random seed for initialization") public Random initRandom = new Random(1);
  @Option(gloss="How much variation in initial parameters") public double initParamsNoise = 0.01;

  @Option(gloss="Regularization for the E-step (important for inconsistent moments)") public double eRegularization = 1e-3;
  @Option(gloss="Regularization for the M-step") public double mRegularization = 0;

  //@Option(gloss="Type of training to use") public ObjectiveType objectiveType = ObjectiveType.unsupervised_gradient;
  //@Option(gloss="Type of optimization to use") public boolean useLBFGS = true;
  @Option(gloss="Use expected measurements (with respect to true distribution)") public boolean expectedMeasurements = true;
  @Option(gloss="Include each (true) measurement with this prob") public double measurementProb = 1;
  @Option(gloss="Number of iterations before switching to full unsupervised") public int numMeasurementIters = Integer.MAX_VALUE;

  @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
  @OptionSet(name="backtrack") public BacktrackingLineSearch.Options backtrack = new BacktrackingLineSearch.Options();

  /**
   * Stores the true counts to be used to print error statistics
   */
  class Analysis {
    Model model;
    Hypergraph<Example> Hp;
    ParamsVec trueParams; // \theta^*
    ParamsVec trueCounts; // E_{\theta^*}[ \phi(X) ]

    /**
     * Initializes the analysis object with true values
     */
    public Analysis( Model model, ParamsVec trueParams ) {
      this.model = model;
      this.trueParams = trueParams;
      this.trueCounts = model.newParamsVec();

      this.Hp = model.createHypergraph(null, trueParams.weights, trueCounts.weights, 1);
      Hp.computePosteriors(false);
      Hp.fetchPosteriors(false);

      // Write to file
      trueParams.write(Execution.getFile("true.params"));
      trueCounts.write(Execution.getFile("true.counts"));
    }

    /**
     * Reports error between estimated parameters and true parameters on
     * the selected fields
     */
    public double reportParams(ParamsVec estimatedParams, boolean[] measuredFeatures) {
      double err = estimatedParams.computeDiff( trueParams, measuredFeatures, null );
      LogInfo.logsForce("paramsError="+err);
      return err;
    }
    public double reportParams(ParamsVec estimatedParams) {
      boolean[] allMeasuredFeatures = new boolean[estimatedParams.numFeatures];
      Arrays.fill( allMeasuredFeatures, true );

      return reportParams(estimatedParams, allMeasuredFeatures);
    }

    /**
     * Reports error between estimated moments and true moments on
     * the selected fields
     */
    public double reportCounts(ParamsVec estimatedCounts, boolean[] measuredFeatures) {
      double err = estimatedCounts.computeDiff( trueCounts, measuredFeatures, null );
      LogInfo.logsForce("countsError="+err);
      return err;
    }
    public double reportCounts(ParamsVec estimatedCounts) {
      boolean[] allMeasuredFeatures = new boolean[estimatedCounts.numFeatures];
      Arrays.fill( allMeasuredFeatures, true );

      return reportParams(estimatedCounts, allMeasuredFeatures);
    }
  }
  public Analysis analysis;

  ////////////////////////////////////////////////////////
  
  // Algorithm parameters
  Model model;

  /**
   * Unrolls data along bottleneck nodes and uses method of moments
   * to return expected potentials
   */
  ParamsVec solveBottleneck( Model model, List<Example> data ) {
    return model.newParamsVec();
  }

  /**
   * Potentially uses moments to find an initial set of parameters that
   * match it. 
   *  - Solved as a restricted M-step, minimizing $\theta^T \tau -
   *    A(\theta)$ or matching $E_\theta[\phi(x) = \tau$.
   */
  ParamsVec initializeParameters(ParamsVec moments, boolean[] measuredFeatures) {
    return model.newParamsVec();
  }
  ParamsVec initializeParameters(ParamsVec moments) {
    return initializeParameters( moments, new boolean[moments.numFeatures] );
  }
  ParamsVec initializeParameters() {
    return initializeParameters( model.newParamsVec() );
  }

  /**
   * Solves EM for the model using data and initial parameters.
   */
  ParamsVec solveEM(List<Example> data, ParamsVec initialParams) {
    return initialParams;
  }

  /**
   * Uses method of moments to find moments along bottlenecks,
   * initializes parameters using them, and runs EM.
   */
  ParamsVec solveBottleneckEM( List<Example> data ) {
    return model.newParamsVec();
  }

  public enum ModelType { mixture, hmm, tallMixture, grid, factMixture };
  public static class ModelOptions {
    @Option(gloss="Type of model") public ModelType modelType = ModelType.mixture;
    @Option(gloss="Number of values of the hidden variable") public int K = 3;
    @Option(gloss="Number of possible values of output") public int D = 5;
    @Option(gloss="Length of observation sequence") public int L = 3;
  }
  @OptionSet(name="model") public ModelOptions modelOpts = new ModelOptions();

  public static class GenerationOptions {
    @Option(gloss="Random seed for generating artificial data") public Random genRandom = new Random(1);
    @Option(gloss="Random seed for the true model") public Random trueParamsRandom = new Random(1);
    @Option(gloss="Number of examples to generate") public int genNumExamples = 100;
    @Option(gloss="How much variation in true parameters") public double trueParamsNoise = 1;
  }
  @OptionSet(name="gen") public GenerationOptions genOpts = new GenerationOptions();;
  PrintWriter eventsOut;

  /**
   * Generates random data from the model.
   *  - Uses genRand as a seed.
   */
  ParamsVec generateParameters( Model model, GenerationOptions opts ) {
    ParamsVec trueParams = model.newParamsVec();
    trueParams.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
    return trueParams;
  }

  /**
   * Generates random data from the model.
   *  - Uses genRand as a seed.
   */
  List<Example> generateData( Model model, ParamsVec params, GenerationOptions opts ) {
    ParamsVec counts = model.newParamsVec();
    Hypergraph<Example> Hp = model.createHypergraph(null, params.weights, counts.weights, 1);
    // Necessary preprocessing before you can generate hyperpaths
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);

    List<Example> examples = new ArrayList<Example>();

    for (int i = 0; i < opts.genNumExamples; i++) {
      Example ex = model.newExample();
      Hp.fetchSampleHyperpath(opts.genRandom, ex);
      examples.add(ex);
      LogInfo.logs("x = %s", Fmt.D(ex.x));
    }

    return examples;
  }

  /**
   * Generates a model of a particular type
   */
  Model generateModel(ModelOptions opts) {
    Model model_;
    switch (opts.modelType) {
      case mixture: {
        MixtureModel model = new MixtureModel();
        model.L = opts.L;
        model.D = opts.D;
        model_ = model;
        break;
      }
      case hmm: {
        HiddenMarkovModel model = new HiddenMarkovModel();
        model.L = opts.L;
        model.D = opts.D;
        model_ = model;
        break;
      }
      case tallMixture: {
        TallMixture model = new TallMixture();
        model.L = opts.L;
        model.D = opts.D;
        model_ = model;
        break;
      }
      case grid: {
        Grid model = new Grid();
        model.width = 2;
        model.height = opts.L/2;
        model.L = opts.L;
        model.D = opts.D;
        model_ = model;
        break;
      }
      default:
        throw new RuntimeException("Unhandled model type: " + opts.modelType);
    }

    model_.K = opts.K;

    // Run once to just instantiate features
    model_.createHypergraph(null, null, null, 0);
    return model_;
  }
  void setModel(Model model) {
    this.model = model;
  }

  public void run() {
    eventsOut = IOUtils.openOutHard(Execution.getFile("events"));

    // Setup; generate model 
    Model model = generateModel( modelOpts );
    setModel( model );

    // Generate parameters
    ParamsVec trueParams = generateParameters( model, genOpts );
    Analysis analysis = new Analysis( model, trueParams );

    // Get true parameters
    List<Example> data = generateData( model, trueParams, genOpts );

    // Run the bottleneck spectral algorithm
    ParamsVec params = solveBottleneckEM(data);

    // Return the error in estimate
    analysis.reportParams( params );
  }

  public static void main(String[] args) {
    Execution.run(args, new BottleneckSpectralEM() );
  }
}


