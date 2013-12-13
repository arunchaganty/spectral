package learning.models.loglinear;

import java.io.IOException;
import java.util.*;

import fig.basic.*;
import fig.exec.*;

import learning.data.ComputableMoments;
import learning.data.HasSampleMoments;
import learning.models.ExponentialFamilyModel;
import learning.models.MixtureOfGaussians;
import learning.spectral.applications.ParameterRecovery;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;
import learning.linalg.*;
import learning.spectral.TensorMethod;
import learning.utils.Counter;

import static learning.models.loglinear.Models.*;
import static learning.Misc.*;
import static learning.models.loglinear.SpectralMeasurements.Mode.*;

/**
 * NIPS 2013
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
    ParamsVec trueParams; // \theta^*
    ParamsVec trueCounts; // E_{\theta^*}[ \phi(X) ]
    Counter<Example> trueMarginal;
    Counter<Example> examples;
    double Zp, perp_p;
    double Zq, perp_q;

    /**
     * Initializes the analysis object with true values
     */
    public Analysis( ExponentialFamilyModel<Example> model, ParamsVec trueParams, Counter<Example> examples ) {
      this.model = model;
      this.trueParams = trueParams;
      this.trueCounts = model.newParamsVec();
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
    public double reportParams(ParamsVec estimatedParams) {
      estimatedParams.write(Execution.getFile("fit.params"));

      Zq = model.getLogLikelihood(estimatedParams);
      ParamsVec estimatedCounts = model.getMarginals(estimatedParams);
      perp_q = -(model.getLogLikelihood(estimatedParams, examples) - Zq);

      Counter<Example> fitMarginal = model.getDistribution(estimatedParams);

      LogInfo.logsForce("true-perp="+perp_p);
      Execution.putOutput("fit-perp", perp_q);
      LogInfo.logsForce("fit-perp="+perp_q);
      LogInfo.logsForce("perp-error="+ Math.abs(perp_p - perp_q));

      // Write to file
      estimatedCounts.write(Execution.getFile("fit.counts"));
      IOUtils.writeLinesHard(Execution.getFile("fit.marginal"), Collections.<String>singletonList(fitMarginal.toString()));

      double err = estimatedCounts.computeDiff( trueCounts, new int[trueParams.K] );
      Execution.putOutput("countsError", err);
      LogInfo.logsForce("countsError="+err);

      err = estimatedParams.computeDiff( trueParams, new int[trueParams.K] );
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

  ParamsVec computeTrueMeasurements( final Counter<Example> data ) {
    assert( analysis != null );

    // Choose the measured features.
    Indexer<Feature> measuredFeatureIndexer = new Indexer<>();
    // This is an initialization of sorts
    Indexer<Feature> features = modelA.newParamsVec().featureIndexer;
    int numFeatures = modelA.numFeatures();
    List<Integer> perm = RandomFactory.permutation(numFeatures, initRandom);

    for(int i = 0; i < Math.round(measuredFraction * numFeatures); i++) {
      Feature feature = features.getObject(perm.get(i));
      measuredFeatureIndexer.add(feature);
    }
    // Now set the measurements to be the true counts
    ParamsVec measurements = new ParamsVec(modelB.getK(), measuredFeatureIndexer);
    ParamsVec.project(analysis.trueCounts, measurements);

    // Add noise
    if( trueMeasurementNoise > 0. ) {
      // Anneal the noise at a 1/sqrt(n) rate.
      trueMeasurementNoise = trueMeasurementNoise / Math.sqrt(data.sum());
      for(int i = 0; i < measurements.weights.length; i++)
        measurements.weights[i] += RandomFactory.randn(trueMeasurementNoise);
    }

    return measurements;
  }

  /**
   * Unrolls data along bottleneck nodes and uses method of moments
   * to return expected potentials
   */
  ParamsVec computeSpectralMeasurements( final Counter<Example> data ) {
    LogInfo.begin_track("solveBottleneck");
    ParamsVec measurements;
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
        for( int d = 0; d < D; d++ ) {
          // Assuming identical distribution.
          measuredFeatureIndexer.add( new UnaryFeature(h, "x="+d) );
        }
      }

      measurements = new ParamsVec(model.K, measuredFeatureIndexer);

      for( int h = 0; h < K; h++ ) {
        for( int d = 0; d < D; d++ ) {
          // Assuming identical distribution.
          int f = measurements.featureIndexer.getIndex(new UnaryFeature(h, "x="+d));
          // multiplying by pi to go from E[x|h] -> E[x,h]
          // multiplying by L because true.counts aggregates
          // over x1, x2 and x3.
          measurements.weights[f] = model.L * M3.get( d, h ) * pi.get(h);
        }
      }
      Execution.putOutput("moments.params", MatrixFactory.fromVector(measurements.weights));
    } else if( modelA instanceof  HiddenMarkovModel || modelA instanceof UndirectedHiddenMarkovModel ) {
//      HiddenMarkovModel model = (HiddenMarkovModel) modelA;
      UndirectedHiddenMarkovModel model = (UndirectedHiddenMarkovModel) modelA;
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
        for( int d = 0; d < D; d++ ) {
          // O
          measuredFeatureIndexer.add( new UnaryFeature(h, "x="+d) );
        }
      }

      measurements = new ParamsVec(model.K, measuredFeatureIndexer);

      for( int h = 0; h < K; h++ ) {
        for( int d = 0; d < D; d++ ) {
          // Assuming identical distribution.
          int f = measurements.featureIndexer.getIndex(new UnaryFeature(h, "x="+d));
          // multiplying by pi to go from E[x|h] -> E[x,h]
          // multiplying by L because true.counts aggregates
          // over x1, x2 and x3.
          measurements.weights[f] = model.L * O.get( d, h ) * pi.get(h);
        }
//        if( useTransitions ) {
//          for( int h_ = 0; h_ < K; h_++ ) {
//            // Assuming identical distribution.
//            int f = measurements.featureIndexer.getIndex(new BinaryFeature(h, h_));
//            // multiplying by pi to go from E[x|h] -> E[x,h]
//            // multiplying by L because true.counts aggregates
//            // over x1, x2 and x3.
//            measurements.weights[f] = (model.L-1) * T.get( h_, h ) * pi.get(h);
//          }
//        }
      }
      Execution.putOutput("moments.params", MatrixFactory.fromVector(measurements.weights));
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

      // measurements.weights[ measurements.featureIndexer.getIndex(Feature("h=0,x=0"))] = 0.0;
      Indexer<Feature> measuredFeatureIndexer = new Indexer<>();
      for( int h = 0; h < K; h++ ) {
        for( int d = 0; d < D; d++ ) {
          // O
          measuredFeatureIndexer.add( new UnaryFeature(h, "x="+d) );
        }
      }

      measurements = new ParamsVec(model.K, measuredFeatureIndexer);

      for( int h = 0; h < K; h++ ) {
        for( int d = 0; d < D; d++ ) {
          // Assuming identical distribution.
          int f = measurements.featureIndexer.getIndex(new UnaryFeature(h, "x="+d));
          // multiplying by pi to go from E[x|h] -> E[x,h]
          // multiplying by L because true.counts aggregates
          // over x1, x2 and x3.
          measurements.weights[f] = model.L * O.get( d, h ) * pi.get(h);
        }
      }
      Execution.putOutput("moments.params", MatrixFactory.fromVector(measurements.weights));

    } else {
      throw new RuntimeException("Not implemented yet");
    }
    LogInfo.end_track("solveBottleneck");

    return measurements;
  }

//  /**
//   * Uses the params and modelA to label the hidden labels for each
//   * data point and reports Hamming errors.
//   */
//  @SuppressWarnings("unchecked")
//  public double reportAccuracy( ParamsVec params, List<Example> examples ) {
//    LogInfo.begin_track("report-accuracy");
//    ParamsVec counts = modelA.newParamsVec();
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

  public enum ModelType { mixture, hmm, tallMixture, grid, factMixture };
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
    @Option(gloss="Number of examples to generate") public int genNumExamples = 100;
    @Option(gloss="How much variation in true parameters") public double trueParamsNoise = 1.0;
  }
  @OptionSet(name="gen") public GenerationOptions genOpts = new GenerationOptions();;

  /**
   * Generates random data from the modelA.
   *  - Uses genRand as a seed.
   */
  ParamsVec generateParameters( ExponentialFamilyModel<Example> model, GenerationOptions opts ) {
    ParamsVec trueParams = model.newParamsVec();
    trueParams.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
//    for(int i = 0; i < trueParams.weights.length; i++)
//      trueParams.weights[i] = Math.sin(i);
    return trueParams;
  }

  /**
   * Generates random data from the modelA.
   *  - Uses genRand as a seed.
   */
  Counter<Example> generateData(Model model, ParamsVec params, GenerationOptions opts) {
    LogInfo.begin_track("generateData");
    ParamsVec counts = model.newParamsVec();
    Hypergraph<Example> Hp = model.createHypergraph(params.weights, counts.weights, 1);
    // Necessary preprocessing before you can generate hyperpaths
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);

    Counter<Example> examples = new Counter<>();

    for (int i = 0; i < opts.genNumExamples; i++) {
      Example ex = model.newExample();
      Hp.fetchSampleHyperpath(opts.genRandom, ex);
      examples.add(ex);
      if( debug ) {
        LogInfo.logs("x = %s", Fmt.D(ex.x));
        LogInfo.logs("h = %s", Fmt.D(ex.h));
      }
    }
    LogInfo.end_track();

    return examples;
  }

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
        modelA = new UndirectedHiddenMarkovModel(opts.K, opts.D, opts.L);
        modelB = new UndirectedHiddenMarkovModel(opts.K, opts.D, opts.L);
//        modelA = new HiddenMarkovModel(opts.K, opts.D, opts.L);
//        modelB = new HiddenMarkovModel(opts.K, opts.D, opts.L);
        break;
      }
      case tallMixture: {
        throw new RuntimeException("Tall mixture not implemented");
        //break;
      }
      case grid: {
        modelA = new GridModel(opts.K, opts.D, opts.L);
        modelB = new GridModel(opts.K, opts.D, opts.L);
        break;
      }
      default:
        throw new RuntimeException("Unhandled modelA type: " + opts.modelType);
    }
  }

  public void run() {
    // Setup modelA, modelB
    initializeModels( modelOpts );

    // Generate parameters
    ParamsVec trueParams = generateParameters( modelA, genOpts );

    // Get true parameters
    Counter<Example> data = modelA.drawSamples(trueParams, genOpts.genRandom, genOpts.genNumExamples);

    analysis = new Analysis( modelA, trueParams, data );

    ParamsVec params = new ParamsVec(analysis.trueParams);
    if(!initializeWithExact)
      params.initRandom(initRandom, initParamsNoise);
    else {
      ParamsVec noise = modelA.newParamsVec();
      noise.initRandom(initRandom, initParamsNoise);
      params.incr(Math.sqrt(0.01), noise);
    }

    // Run the bottleneck spectral algorithm
    switch( mode ) {
      case EM:
        params = emSolver.solveEM(modelA, data, params);
        break;
      case TrueMeasurements: {
        ParamsVec bottleneckMeasurements = computeTrueMeasurements( data );
        bottleneckMeasurements.write(Execution.getFile("measurements.counts"));
        ParamsVec beta = new ParamsVec(bottleneckMeasurements);
        beta.clear();
        // Use these measurements to solve for parameters
        params = measurementsEMSolver.solveMeasurements(
                modelA, modelB, data, bottleneckMeasurements, params, beta).getFirst();
        } break;
      case SpectralMeasurements: {
        ParamsVec bottleneckMeasurements = computeSpectralMeasurements( data );
        bottleneckMeasurements.write(Execution.getFile("measurements.counts"));
        ParamsVec beta = new ParamsVec(bottleneckMeasurements);
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


