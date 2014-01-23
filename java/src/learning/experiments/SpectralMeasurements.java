package learning.experiments;

import fig.basic.*;
import fig.exec.Execution;
import learning.common.Counter;
import learning.data.ComputableMoments;
import learning.data.HasSampleMoments;
import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;
import learning.models.BasicParams;
import learning.models.ExponentialFamilyModel;
import learning.models.MixtureOfGaussians;
import learning.models.Params;
import learning.models.loglinear.*;
import learning.models.loglinear.Models.GridModel;
import learning.models.loglinear.Models.MixtureModel;
import learning.spectral.TensorMethod;
import learning.spectral.applications.ParameterRecovery;
import learning.unsupervised.ExpectationMaximization;
import learning.unsupervised.MeasurementsEM;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

import java.util.Collections;
import java.util.List;
import java.util.Random;

import static learning.experiments.SpectralMeasurements.Mode.TrueMeasurements;
import static learning.models.loglinear.Models.*;

/**
 * ICML 2014
 * Uses method of moments to initialize parameters 
 * for EM for general log-linear models.
 *  - Uses Hypergraph framework to represent a modelA.
 */
public class SpectralMeasurements implements Runnable {
  public static enum Mode {
    EM,
    TrueMeasurements,
    SpectralMeasurements
  }
  @Option(gloss="Run mode") public Mode mode = TrueMeasurements;

  @OptionSet(name="MeasurementsEM") public MeasurementsEM measurementsEMSolver = new MeasurementsEM();
  @OptionSet(name="EMSolver") public ExpectationMaximization emSolver = new ExpectationMaximization();

  @Option(gloss="Random seed for initialization") public Random initRandom = new Random(1);
  @Option(gloss="How much variation in initial parameters") public double initParamsNoise = 1.0;

  //@Option(gloss="Type of training to use") public ObjectiveType objectiveType = ObjectiveType.unsupervised_gradient;
  @Option(gloss="Include each (true) measurement with this prob") public double measuredFraction = 1.;
  @Option(gloss="Include gaussian noise with this variance to true measurements") public double trueMeasurementNoise = 0.0;

  @Option(gloss="Initialize with exact") public boolean infiniteSamples = false;

  @Option(gloss="Initialize with exact") public boolean initializeWithExact = false;
  @Option(gloss="Preconditioning") public double preconditioning = 0.0;
  @Option(gloss="Smooth measurements") public double smoothMeasurements = 0.0;

  @Option(gloss="Use T in SpectralMeasurements?") public boolean useTransitions = false;
  @Option(gloss="HMM window size") public int windowSize = 1;

  @OptionSet(name="tensor") public TensorMethod algo = new TensorMethod();

  @Option(gloss="Print all the things") public boolean debug = false;

  /**
   * Stores the true counts to be used to print error statistics
   */
  class Analysis {
    ExponentialFamilyModel<Example> model;
    Params trueParams; // \theta^*
    Params trueCounts; // E_{\theta^*}[ \phi(X) ]
    Counter<Example> trueMarginal;
    Counter<Example> examples;
    double Zp, perp_p;
    double Zq, perp_q;

    /**
     * Initializes the analysis object with true values
     */
    public Analysis( ExponentialFamilyModel<Example> model, Params trueParams, Counter<Example> examples ) {
      this.model = model;
      this.trueParams = trueParams;
      this.trueCounts = model.newParams();
      this.examples = examples;

      Zp = model.getLogLikelihood(trueParams);
      trueCounts = model.getMarginals(trueParams);
      perp_p = -(model.getLogLikelihood(trueParams,examples) - Zp);
      Execution.putOutput("true-perp", perp_p);
      LogInfo.logsForce("true-perp=" + perp_p);

      trueMarginal = model.getDistribution(trueParams);

      // Write to file
      trueParams.write(Execution.getFile("true.params"));
      trueCounts.write(Execution.getFile("true.counts"));
      IOUtils.writeLinesHard(Execution.getFile("true.marginal"), Collections.<String>singletonList(trueMarginal.toString()));
    }

    /**
     * Reports error between estimated parameters and true parameters on
     * the selected fields
     */
    public double reportParams(Params estimatedParams) {
      estimatedParams.write(Execution.getFile("fit.params"));

      Zq = model.getLogLikelihood(estimatedParams);
      Params estimatedCounts = model.getMarginals(estimatedParams);
      perp_q = -(model.getLogLikelihood(estimatedParams, examples) - Zq);

      Counter<Example> fitMarginal = model.getDistribution(estimatedParams);

      LogInfo.logsForce("true-perp="+perp_p);
      Execution.putOutput("fit-perp", perp_q);
      LogInfo.logsForce("fit-perp="+perp_q);
      LogInfo.logsForce("perp-error="+ Math.abs(perp_p - perp_q));

      // Write to file
      estimatedCounts.write(Execution.getFile("fit.counts"));
      IOUtils.writeLinesHard(Execution.getFile("fit.marginal"), Collections.<String>singletonList(fitMarginal.toString()));

      double err;
      int K = estimatedCounts.numGroups();
      int[] perm = new int[K];
      err = estimatedCounts.computeDiff( trueCounts, perm);
      Execution.putOutput("countsError", err);
      LogInfo.logsForce("countsError="+err);

      err = estimatedParams.computeDiff( trueParams, perm );
      Execution.putOutput("paramsError", err);
      LogInfo.logsForce("paramsError="+err);

      err = Counter.diff(trueMarginal, fitMarginal);
      Execution.putOutput("marginalError", err);
      LogInfo.logsForce("marginalError="+err);

      return err;
    }

  }
  public Analysis analysis;

  // Algorithm parameters
  ExponentialFamilyModel<Example> modelA;
  ExponentialFamilyModel<Example> modelB;

  /**
   * Computes moments from the sequences in an Example.
   */
  class ExampleMoments implements ComputableMoments, HasSampleMoments {
    //List<Integer> indices;
    SimpleMatrix P13;
    SimpleMatrix P12;
    SimpleMatrix P32;
    FullTensor P123;

    <T> ExampleMoments(ExponentialFamilyModel<T> model, final Counter<T> data) {
      int D = model.getD();
      // Create P13
      P13 = new SimpleMatrix(D, D);
      P12 = new SimpleMatrix(D, D);
      P32 = new SimpleMatrix(D, D);
      P123 = new FullTensor(D,D,D);
      double count = 0;
      for( T ex : data ) {
        // IMPORTANT: This automagically makes what we want x3, which should be the most stable thing.
        count += model.updateMoments(ex, data.getCount(ex), P12, P13, P32, P123);
      }
      // Scale down everything
      P13 = P13.scale(1./count);
      P12 = P12.scale(1./count);
      P32 = P32.scale(1./count);
      P123.scale(1./count);

      // Add some to the diagonal term
      if( preconditioning > 0. ) {
        for( int d = 0; d < D; d++ ) {
          P123.set(d,d,d, P123.get(d,d,d) + preconditioning);
        }
      }
      P123.scale(1./P123.elementSum());
    }

    @Override
    public MatrixOps.Matrixable computeP13() {
      return MatrixOps.matrixable(P13);
    }

    @Override
    public MatrixOps.Matrixable computeP12() {
      return MatrixOps.matrixable(P12);
    }

    @Override
    public MatrixOps.Matrixable computeP32() {
      return MatrixOps.matrixable(P32);
    }

    @Override
    public MatrixOps.Tensorable computeP123() {
      return MatrixOps.tensorable(P123);
    }

    @Override
    public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> computeSampleMoments(int N) {
      return Quartet.with(P13, P12, P32, P123);
    }
  }

  Params computeTrueMeasurements( final Counter<Example> data ) {
    assert( analysis != null );
    int K = modelOpts.K;

    // Choose the measured features.
    Indexer<Feature> measuredFeatureIndexer = new Indexer<>();
    // This is an initialization of sorts
    Indexer<Feature> features = modelA.newParams().getFeatureIndexer();
    int numFeatures = modelA.numFeatures();
    List<Integer> perm = RandomFactory.permutation(numFeatures, initRandom);

    for(int i = 0; i < Math.round(measuredFraction * numFeatures); i++) {
      Feature feature = features.getObject(perm.get(i));
      measuredFeatureIndexer.add(feature);
    }
    // Now set the measurements to be the true counts
    Params measurements = new BasicParams(K, measuredFeatureIndexer);
    measurements.copyOver(analysis.trueCounts);

    // Add noise
    if( trueMeasurementNoise > 0. ) {
      // Anneal the noise at a 1/sqrt(n) rate.
      trueMeasurementNoise = trueMeasurementNoise / Math.sqrt(data.sum());
      for(int i = 0; i < measurements.size(); i++)
        measurements.toArray()[i] += RandomFactory.randn(initRandom, trueMeasurementNoise);
    }

    return measurements;
  }

  /**
   * Unrolls data along bottleneck nodes and uses method of moments
   * to return expected potentials
   */
  Params computeSpectralMeasurements( final Counter<Example> data ) {
    LogInfo.begin_track("solveBottleneck");
    Params measurements;
    // Construct triples of three observed variables around the hidden
    // node.
    int K = modelA.getK(); int D = modelA.getD();

    if( modelA instanceof  MixtureModel ) {
      MixtureModel model = (MixtureModel) modelA;
      // \phi_1, \phi_2, \phi_3

      MixtureOfGaussians gmm = ParameterRecovery.recoverGMM(K, 0, new ExampleMoments(model, data), smoothMeasurements);
      // TODO: Create a mixture of Bernoullis and move this stuff there.
      SimpleMatrix pi = gmm.getWeights();
      SimpleMatrix[] M = gmm.getMeans();
      for(int i = 0; i < model.L; i++ )
        M[i] = MatrixOps.projectOntoSimplex( M[i], smoothMeasurements );
      // Average the three Ms
      SimpleMatrix M3 = (M[0].plus(M[1]).plus(M[2])).scale(1.0/3.0);
      //SimpleMatrix M3 = M[2]; // M3 is most accurate.
      M3 = MatrixOps.projectOntoSimplex( M3, smoothMeasurements );

      // measurements.weights[ measurements.featureIndexer.getIndex(Feature("h=0,x=0"))] = 0.0;
      Indexer<Feature> measuredFeatureIndexer = new Indexer<>();
      for( int h = 0; h < K; h++ ) {
        for( int x = 0; x < D; x++ ) {
          // Assuming identical distribution.
          measuredFeatureIndexer.add(new UnaryFeature(h, "x="+x));
        }
      }
      measurements = new BasicParams(K, measuredFeatureIndexer);

      for( int h = 0; h < K; h++ ) {
        for( int x = 0; x < D; x++ ) {
          // Assuming identical distribution.
          // multiplying by pi to go from E[x|h] -> E[x,h]
          // multiplying by L because true.counts aggregates
          // over x1, x2 and x3.
          measurements.set(new UnaryFeature(h, "x="+x), model.L * M3.get( x, h ) * pi.get(h));
        }
      }
    } else if( modelA instanceof HiddenMarkovModel) {
      HiddenMarkovModel model = (HiddenMarkovModel) modelA;
      // \phi_1, \phi_2, \phi_3

      MixtureOfGaussians gmm = ParameterRecovery.recoverGMM(K, 0, new ExampleMoments(model, data), smoothMeasurements);
      // TODO: Create a mixture of Bernoullis and move this stuff there.
      SimpleMatrix pi = gmm.getWeights();
      // The pi are always really bad. Ignore?
      SimpleMatrix[] M = gmm.getMeans();
      SimpleMatrix O = M[2];

      Execution.putOutput("pi", pi);
      Execution.putOutput("O", O);

//        pi = MatrixFactory.ones(K).scale(1./K);

      // Project onto simplices
      O = MatrixOps.projectOntoSimplex( O, smoothMeasurements );

      Indexer<Feature> measuredFeatureIndexer = new Indexer<>();
      for( int h = 0; h < K; h++ ) {
        for( int x = 0; x < D; x++ ) {
          // O
          measuredFeatureIndexer.add(new UnaryFeature(h, "x="+x));
        }
      }

      measurements = new BasicParams(K, measuredFeatureIndexer);

      for( int h = 0; h < K; h++ ) {
        for( int x = 0; x < D; x++ ) {
          // Assuming identical distribution.
          // multiplying by pi to go from E[x|h] -> E[x,h]
          // multiplying by L because true.counts aggregates
          // over x1, x2 and x3.
          measurements.set(new UnaryFeature(h, "x="+x), model.L * O.get( x, h ) * pi.get(h));
        }
      }
    } else if( modelA instanceof  GridModel ) {
      GridModel model = (GridModel) modelA;
      // \phi_1, \phi_2, \phi_3

      MixtureOfGaussians gmm = ParameterRecovery.recoverGMM(K, 0, new ExampleMoments(model, data), smoothMeasurements);
      SimpleMatrix pi = gmm.getWeights();
      // The pi are always really bad. Ignore?
      SimpleMatrix[] M = gmm.getMeans();
      SimpleMatrix O = M[2];
      SimpleMatrix O_ = M[1];

      Execution.putOutput("pi", pi);
      Execution.putOutput("O", O);

      // Average the two
      O = O.plus(O_).scale(0.5);

      // Project onto simplices
      O = MatrixOps.projectOntoSimplex( O, smoothMeasurements );
      Execution.putOutput("O", O);

      Indexer<Feature> measuredFeatureIndexer = new Indexer<>();
      for( int h = 0; h < K; h++ ) {
        for( int x = 0; x < D; x++ ) {
          // O
          measuredFeatureIndexer.add(new UnaryFeature(h, "x="+x));
        }
      }

      measurements = new BasicParams(K, measuredFeatureIndexer);

      for( int h = 0; h < K; h++ ) {
        for( int x = 0; x < D; x++ ) {
          // Assuming identical distribution.
          // multiplying by pi to go from E[x|h] -> E[x,h]
          // multiplying by L because true.counts aggregates
          // over x1, x2 and x3.
          measurements.set(new UnaryFeature(h, "x="+x), 2 * model.L * O.get( x, h ) * pi.get(h));
        }
      }
    } else {
      throw new RuntimeException("Not implemented yet");
    }
    Execution.putOutput("moments.params", Fmt.D(measurements.toArray()));
    LogInfo.end_track("solveBottleneck");

    return measurements;
  }

//  /**
//   * Uses the params and modelA to label the hidden labels for each
//   * data point and reports Hamming errors.
//   */
//  @SuppressWarnings("unchecked")
//  public double reportAccuracy( Params params, List<Example> examples ) {
//    LogInfo.begin_track("report-accuracy");
//    Params counts = modelA.newParamsVec();
//    int K = modelA.getK();
//
//    // Create a confusion matrix with prediced vs estimated label choices
//    double[][] labelMapping = new double[K][K];
//    for(int i = 0; i < examples.size(); i++ ) {
//      Example ex = examples.get(i);
//      // Create a copy of the example with just the data; use this to
//      // guess the labels.
//      Example ex_ = ex.copyData();
//      // Cache the hypergraph
//      Hypergraph Hq = modelA.createHypergraph(ex_, params.weights, counts.weights, 1.0/examples.size());
//      Hq.computePosteriors(false);
//      Hq.fetchPosteriors(false); // Places the posterior expectation $E_{Y|X}[\phi]$ into counts
//      // Fill in the ex_.h values
//      Hq.fetchBestHyperpath(ex_);
//
//      for( int l = 0; l < ex.h.length; l++ )
//        labelMapping[ex.h[l]][ex_.h[l]] += 1;
//    }
//    //if( debug )
//      LogInfo.dbg( "Label mapping: \n" + Fmt.D( labelMapping ) );
//    double acc = bestAccuracy( labelMapping );
//    LogInfo.end_track();
//
//    return acc;
//  }

  ///////////////////////////////////
  // Instantiation stuff

  public enum ModelType { mixture, hmm, tallMixture, grid, factMixture }

  public static class ModelOptions {
    @Option(gloss="Type of modelA") public ModelType modelType = ModelType.mixture;
    @Option(gloss="Number of values of the hidden variable") public int K = 3;
    @Option(gloss="Number of possible values of output") public int D = 5;
    @Option(gloss="Length of observation sequence") public int L = 3;
  }
  @OptionSet(name="modelA") public ModelOptions modelOpts = new ModelOptions();

  public static class GenerationOptions {
    @Option(gloss="Random seed for generating artificial data") public Random genRandom = new Random(42);
    @Option(gloss="Random seed for the true modelA") public Random trueParamsRandom = new Random(43);
    @Option(gloss="Number of examples to generate") public double genNumExamples = 100;
    @Option(gloss="How much variation in true parameters") public double trueParamsNoise = 1.0;
  }
  @OptionSet(name="gen") public GenerationOptions genOpts = new GenerationOptions();

  /**
   * Generates a modelA of a particular type
   */
  void initializeModels(ModelOptions opts) {
    switch (opts.modelType) {
      case mixture: {
        modelA = new MixtureModel(opts.K, opts.D, opts.L);
        modelB = new MixtureModel(opts.K, opts.D, opts.L);
        break;
      }
      case hmm: {
        modelA = new HiddenMarkovModel(opts.K, opts.D, opts.L);
        modelB = new HiddenMarkovModel(opts.K, opts.D, opts.L);
        break;
      }
      case grid: {
//        modelA = new GridModel(opts.K, opts.D, opts.L);
//        modelB = new GridModel(opts.K, opts.D, opts.L);
        modelA = new LatentGridModel(opts.K, opts.D, opts.L);
        modelB = new LatentGridModel(opts.K, opts.D, opts.L);
        break;
      }
      case tallMixture: {
        throw new RuntimeException("Tall mixture not implemented");
        //break;
      }
      default:
        throw new RuntimeException("Unhandled modelA type: " + opts.modelType);
    }
  }

  public void run() {
    // Setup modelA, modelB
    initializeModels( modelOpts );

    // Generate parameters
    Params trueParams = modelA.newParams();
    trueParams.initRandom(genOpts.trueParamsRandom, genOpts.trueParamsNoise);

    // Get true parameters
    Counter<Example> data;
    if(genOpts.genNumExamples > 1e7 || infiniteSamples)
      data = modelA.getDistribution(trueParams);
    else
      data = modelA.drawSamples(trueParams, genOpts.genRandom, (int)genOpts.genNumExamples);

    // Setup analysis
    analysis = new Analysis( modelA, trueParams, data );

    Params params = analysis.trueParams.copy();
    if(!initializeWithExact)
      params.initRandom(initRandom, initParamsNoise);
    else {
      Params noise = modelA.newParams();
      noise.initRandom(initRandom, 0.1);
      params.plusEquals(1.0, noise);
    }

    // Run the bottleneck spectral algorithm
    switch( mode ) {
      case EM:
        params = emSolver.solveEM(modelA, data, params);
        break;
      case TrueMeasurements: {
        Params bottleneckMeasurements = computeTrueMeasurements( data );
        bottleneckMeasurements.write(Execution.getFile("measurements.counts"));
        Params beta = bottleneckMeasurements.copy();
        beta.clear();
        // Use these measurements to solve for parameters
        params = measurementsEMSolver.solveMeasurements(
                modelA, modelB, data, bottleneckMeasurements, params, beta).getFirst();
        } break;
      case SpectralMeasurements: {
        Params bottleneckMeasurements = computeSpectralMeasurements( data );
        bottleneckMeasurements.write(Execution.getFile("measurements.counts"));
        Params beta = bottleneckMeasurements.copy();
        beta.clear();
        // Use these measurements to solve for parameters
        params = measurementsEMSolver.solveMeasurements(
                modelA, modelB, data, bottleneckMeasurements, params, beta).getFirst();
        } break;
    }

    // Return the error in estimate
    analysis.reportParams( params );
  }

  public static void main(String[] args) {
    Execution.run(args, new SpectralMeasurements() );
  }
}


