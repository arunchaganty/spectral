package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;

import static fig.basic.LogInfo.*;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;
import learning.linalg.*;
import learning.spectral.TensorMethod;

import static learning.models.loglinear.Models.*;
import static learning.Misc.*;

/**
 * NIPS 2013
 * Uses method of moments to initialize parameters 
 * for EM for general log-linear models.
 *  - Uses Hypergraph framework to represent a modelA.
 */
public class SpectralMeasurements implements Runnable {
  @OptionSet(name="Measurements") public Measurements measurementsSolver = new Measurements();

  @Option(gloss="Random seed for initialization") public Random initRandom = new Random(1);
  @Option(gloss="How much variation in initial parameters") public double initParamsNoise = 0.01;

  //@Option(gloss="Type of training to use") public ObjectiveType objectiveType = ObjectiveType.unsupervised_gradient;
  @Option(gloss="Use expected measurements (with respect to true distribution)") public boolean expectedMeasurements = true;
  @Option(gloss="Include each (true) measurement with this prob") public double measurementProb = 1;

  @Option(gloss="Smooth measurements") public double smoothMeasurements = 0.0;
  @Option(gloss="Use T in SpectralMeasurements?") public boolean useTransitions = true;
  @OptionSet(name="tensor") public TensorMethod algo = new TensorMethod();

  @Option(gloss="Print all the things") public boolean debug = false;

  Map<Integer,Integer> getCountProfile(List<Example> examples) {
      Map<Integer,Integer> counter = new HashMap<Integer,Integer>();
      for( Example ex: examples ) {
        int L = ex.x.length;
        if( counter.get( L ) == null ) {
          counter.put( L, 1 );
        } else {
          counter.put( L, counter.get(L) + 1 );
        }
      }
      return counter;
  }

  /**
   * Stores the true counts to be used to print error statistics
   */
  class Analysis {
    Model model;
    Map<Integer,Hypergraph<Example>> Hp;
    ParamsVec trueParams; // \theta^*
    ParamsVec trueCounts; // E_{\theta^*}[ \phi(X) ]
    List<Example> examples;

    /**
     * Initializes the analysis object with true values
     */
    public Analysis( Model model, ParamsVec trueParams, List<Example> examples ) {
      this.model = model;
      this.trueParams = trueParams;
      this.trueCounts = model.newParamsVec();
      this.examples = examples;

      Hp = new HashMap<Integer, Hypergraph<Example>>();
      for( Map.Entry<Integer,Integer> pair : getCountProfile(examples).entrySet() ) {
        int L = pair.getKey(); int cnt = pair.getValue();
        Hypergraph<Example> H = model.createHypergraph(L, trueParams.weights, trueCounts.weights, (double) cnt/examples.size());
        H.computePosteriors(false);
        H.fetchPosteriors(false);

        Hp.put( L, H );
      }

      // Write to file
      trueParams.write(Execution.getFile("true.params"));
      trueCounts.write(Execution.getFile("true.counts"));
    }

    /**
     * Reports error between estimated parameters and true parameters on
     * the selected fields
     */
    public double reportParams(ParamsVec estimatedParams) {
      double err = estimatedParams.computeDiff( trueParams, new int[model.K] );
      LogInfo.logsForce("paramsError="+err);
      return err;
    }

    /**
     * Reports error between estimated moments and true moments on
     * the selected fields
     * TODO: Don't use boolean[] measuredFeatures again.
     */
    @Deprecated
    public double reportCounts(ParamsVec estimatedCounts, boolean[] measuredFeatures) {
      double err = estimatedCounts.computeDiff( trueCounts, measuredFeatures, null );
      //LogInfo.logsForce("countsError(%s)=%f [%s] - [%s]", Fmt.D(measuredFeatures), err, Fmt.D(estimatedCounts.weights), Fmt.D(trueCounts.weights));
      LogInfo.logsForce("countsError(%.1f %% of features)=%f", 100*(double)MatrixOps.sum(measuredFeatures)/measuredFeatures.length, err );
      return err;
    }
    public double reportCounts(ParamsVec estimatedCounts) {
      boolean[] allMeasuredFeatures = new boolean[estimatedCounts.numFeatures];
      Arrays.fill( allMeasuredFeatures, true );

      return reportCounts(estimatedCounts, allMeasuredFeatures);
    }
  }
  public Analysis analysis;

  // Algorithm parameters
  Model modelA;
  Model modelB;

  /**
   * Model specific unrolling of the data elements to create three-view data.
   * TODO: Move to Model?
   */
  Iterator<double[][]> constructDataSequence( Model model, final List<Example> data ) {
    LogInfo.begin_track("construct-data-sequence");
    Iterator<double[][]> dataSeq;
    if( model instanceof MixtureModel) {
      final MixtureModel mixModel = (MixtureModel) model;
      int K = mixModel.K; int D = mixModel.D;
      assert( mixModel.L == 3 );

      // x_{1,2,3} 
      dataSeq = (new Iterator<double[][]>() {
        Iterator<Example> iter = data.iterator();
        public boolean hasNext() {
          return iter.hasNext();
        }
        public double[][] next() {
          Example ex = iter.next();
          double[][] data = new double[3][mixModel.D]; // Each datum is a one-hot vector
          for( int v = 0; v < 3; v++ ) {
            data[v][ex.x[v]] = 1.0;
          }

          return data;
        }
        public void remove() {
          throw new RuntimeException();
        }
      });
    } else if( model instanceof HiddenMarkovModel) {
      final HiddenMarkovModel hmmModel = (HiddenMarkovModel) model;
      int K = hmmModel.K; int D = hmmModel.D;

      // x_{1,2,3} 
      dataSeq = (new Iterator<double[][]>() {
        Iterator<Example> iter = data.iterator();
        int[] current; int idx = 0; 
        public boolean hasNext() {
          if( current != null && idx < current.length - 2) 
        return true;
          else
        return iter.hasNext();
        }
        public double[][] next() {
          // Walk through every example, and return triples from the
          // examples.
          if( current == null || idx > current.length - 3 ) {
            Example ex = iter.next();
            current = ex.x;
            idx = 0;
          }

          // Efficiently memoize the random projections here.
          double[][] data = new double[3][hmmModel.D]; // Each datum is a one-hot vector
          for( int v = 0; v < 3; v++ ) {
            data[v][current[idx+v]] = 1.0;
          }
          idx++;

          return data;
        }
        public void remove() {
          throw new RuntimeException();
        }
      });
    } else if( model instanceof GridModel) {
      final GridModel gModel = (GridModel) model;
      final int K = gModel.K; final int D = gModel.D;
      final int L = gModel.L;

      // x_{1,2,3} 
      dataSeq = (new Iterator<double[][]>() {
        Iterator<Example> iter = data.iterator();
        Example current; int idx = 0; 
        public boolean hasNext() {
          if( current != null && idx < current.h.length ) 
        return true;
          else
        return iter.hasNext();
        }
        public double[][] next() {
          // Walk through every example, and return triples from the
          // examples.
          if( current == null || idx >= current.h.length ) {
            current = iter.next();
            idx = 0;
          }

          // Efficiently memoize the random projections here.
          double[][] data = new double[3][D]; // Each datum is a one-hot vector
          for( int v = 0; v < 3; v++ ) {
            // VERY subtly constructed; h[idx] is the hidden node, so x[2*idx],
            // x[2*idx+1] are the children. However, the tensor fact. is
            // most stable for x3, so I'm making these the last two nodes.
            // Hate me later.
            data[v][current.x[(2*idx + (2-v)) % current.x.length]] = 1.0;
          }
          idx++;
          return data;
        }
        public void remove() {
          throw new RuntimeException();
        }
      });
    } else {
      throw new RuntimeException("Unhandled modelA type: " + model.getClass() );
    }
    LogInfo.end_track();
    return dataSeq;
  }
  /**
   * Use the moments returned by MoM to estimate counts.
   * TODO: Move to Model?
   */
  void populateFeatures( Model model, 
        Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> bottleneckMoments,
        ParamsVec measurements, boolean[] measuredFeatures ) {
    LogInfo.begin_track("populate-features");

    int K = model.K; int D = model.D;

    double[] pi = MatrixFactory.toVector( bottleneckMoments.getValue0() );
    MatrixOps.projectOntoSimplex( pi, 1.0 + smoothMeasurements );
    SimpleMatrix M[] = {bottleneckMoments.getValue1(), bottleneckMoments.getValue2(), bottleneckMoments.getValue3()};

      LogInfo.logs( Fmt.D( pi ) );
      LogInfo.logs( M[0] );
      LogInfo.logs( M[1] );
      LogInfo.logs( M[2] );
  
    // Set appropriate measuredFeatures to observed moments
    if( model instanceof MixtureModel ) {
      int L = ((MixtureModel)(model)).L; // Extract from the modelA in a clever way.
      assert( M[2].numRows() == D );
      assert( M[2].numCols() == K );
      // Each column corresponds to a particular hidden moment.
      // Project onto the simplex
      
      // Average over the three M's
      for(int i = 0; i < L; i++ )
        M[i] = MatrixOps.projectOntoSimplex( M[i], smoothMeasurements );
      SimpleMatrix M3 = (M[0].plus(M[1]).plus(M[2])).scale(1.0/3.0);
      //SimpleMatrix M3 = M[2]; // M3 is most accurate.
      M3 = MatrixOps.projectOntoSimplex( M3, smoothMeasurements );
      LogInfo.logs( "pi: " + Fmt.D(pi) );
      LogInfo.logs( "M3: " + M3 );

      for( int h = 0; h < K; h++ ) {
        for( int d = 0; d < D; d++ ) {
          // Assuming identical distribution.
          int f = measurements.featureIndexer.getIndex(new UnaryFeature(h, "x="+d));
          measuredFeatures[f] = true;
          // multiplying by pi to go from E[x|h] -> E[x,h]
          // multiplying by 3 because true.counts aggregates
          // over x1, x2 and x3.
          measurements.weights[f] = L * M3.get( d, h ) * pi[h]; 
        }
      }
      Execution.putOutput("moments.params", MatrixFactory.fromVector(measurements.weights));
    } else if( model instanceof HiddenMarkovModel ) {
      int L = ((HiddenMarkovModel)(model)).L; // Extract from the modelA in a clever way.
      // M[1] is O 
      SimpleMatrix O = M[1];
      // M[2] is OT 
      SimpleMatrix T = O.pseudoInverse().mult(M[2]);
      assert( O.numRows() == D ); assert( O.numCols() == K );
      assert( T.numRows() == K ); assert( T.numCols() == K );

      // Each column corresponds to a particular hidden moment.
      // Project onto the simplex
      // Note: We might need to invert the random projection here.
      O = MatrixOps.projectOntoSimplex( O, smoothMeasurements );
      // smooth measurements by adding a little 
      T = MatrixOps.projectOntoSimplex( T, smoothMeasurements ).transpose();
      LogInfo.logs( "pi: " + Fmt.D(pi) );
      LogInfo.logs( "O: " + O );
      LogInfo.logs( "T: " + T );

      double[][] T_ = MatrixFactory.toArray( T );
      double[][] O_ = MatrixFactory.toArray( O.transpose() );

      for( int i = 0; i < K; i ++) {
        assert( MatrixOps.equal( MatrixOps.sum( T_[i] ), 1 ) );
        assert( MatrixOps.equal( MatrixOps.sum( O_[i] ), 1 ) );
      }

      // Put the observed moments back into the counts.
      for( int h = 0; h < K; h++ ) {
        for( int d = 0; d < D; d++ ) {
          int f = measurements.featureIndexer.getIndex(new UnaryFeature(h, "x="+d));
          measuredFeatures[f] = true;
          // multiplying by pi to go from E[x|h] -> E[x,h]
          // multiplying by 3 because true.counts aggregates
          // over x1, x2 and x3.
          measurements.weights[f] = L * O.get( d, h ) * pi[h];
        }
        // 
        // TODO: Experiment with using T.
        if( useTransitions ) {
          for( int h_ = 0; h_ < K; h_++ ) {
            int f = measurements.featureIndexer.getIndex(new BinaryFeature(h,h_));
            measuredFeatures[f] = true;
            // multiplying by pi to go from E[x|h] -> E[x,h]
            // multiplying by 3 because true.counts aggregates
            // over x1, x2 and x3.
            measurements.weights[f] = (L-1) * T.get( h, h_ ) * pi[h]; 
          }
        }
      }
      Execution.putOutput("moments.params", Fmt.D( measurements.weights ));
    } else if( model instanceof GridModel) {
      int L = ((GridModel)(model)).L; // Extract from the modelA in a clever way.
      assert( M[2].numRows() == D );
      assert( M[2].numCols() == K );
      // Each column corresponds to a particular hidden moment.
      // Project onto the simplex
      
      // Average over the last two M's
      for(int i = 1; i < 3; i++ )
        M[i] = MatrixOps.projectOntoSimplex( M[i], smoothMeasurements );
      SimpleMatrix M3 = (M[1]).plus(M[2]).scale(1.0/2.0);
      //SimpleMatrix M3 = M[2]; // M3 is most accurate.
      M3 = MatrixOps.projectOntoSimplex( M3, smoothMeasurements );
      LogInfo.logs( "pi: " + Fmt.D(pi) );
      LogInfo.logs( "M3: " + M3 );

      for( int h = 0; h < K; h++ ) {
        for( int d = 0; d < D; d++ ) {
          // Assuming identical distribution.
          int f = measurements.featureIndexer.getIndex(new UnaryFeature(h, "x="+d));
          measuredFeatures[f] = true;
          // multiplying by pi to go from E[x|h] -> E[x,h]
          // multiplying by 3 because true.counts aggregates
          // over x1, x2 and x3.
          measurements.weights[f] = L * M3.get( d, h ) * pi[h]; 
        }
      }
      Execution.putOutput("moments.params", MatrixFactory.fromVector(measurements.weights));

    } else {
      throw new RuntimeException("Unhandled modelA type: " + model.getClass() );
    }
  
    LogInfo.end_track();
  }
  /**
   * Unrolls data along bottleneck nodes and uses method of moments
   * to return expected potentials
   */
  @SuppressWarnings("unchecked")
  ParamsVec solveBottleneck( final List<Example> data ) {
    LogInfo.begin_track("solveBottleneck");

    ParamsVec measurements = null;
    if( expectedMeasurements ) { // From expected counts
      // Choose the measured features.
      Indexer<Feature> measuredFeatureIndexer = new Indexer<>();
      assert( analysis != null );
      Random random = new Random(3);
      for( Feature f : modelA.featureIndexer ) {
        if( random.nextDouble() < measurementProb )
          measuredFeatureIndexer.add(f);
      }
      // Now set the measurements to be the true counts
      measurements = new ParamsVec(modelB.K, measuredFeatureIndexer);
      ParamsVec.project(analysis.trueCounts, measurements);
    } else { // From spectral methods
      // Construct triples of three observed variables around the hidden
      // node.
      int K = modelA.K; int D = modelA.D;

      throw new RuntimeException("Not implemented yet");

      // TODO: FIXME
//      Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> bottleneckMoments = null;
//      Iterator<double[][]> dataSeq = constructDataSequence(modelA, data );
//      bottleneckMoments = algo.recoverParameters( K, D, dataSeq );
//      populateFeatures(modelA, bottleneckMoments, measurements, measuredFeatures );
    }
    LogInfo.logs("sum_counts: " + MatrixOps.sum(measurements.weights));
    if(analysis != null)  LogInfo.logs("sum_counts (true): " + MatrixOps.sum(analysis.trueCounts.weights));
//    if(analysis != null) analysis.reportCounts( measurements, measuredFeatures );

    LogInfo.end_track("solveBottleneck");

    return measurements;
  }

  /**
   * Uses the params and modelA to label the hidden labels for each
   * data point and reports Hamming errors. 
   */
  @SuppressWarnings("unchecked")
  public double reportAccuracy( ParamsVec params, List<Example> examples ) {
    LogInfo.begin_track("report-accuracy");
    ParamsVec counts = modelA.newParamsVec();
    int K = modelA.K;

    // Create a confusion matrix with prediced vs estimated label choices
    double[][] labelMapping = new double[K][K];
    for(int i = 0; i < examples.size(); i++ ) {
      Example ex = examples.get(i);
      // Create a copy of the example with just the data; use this to
      // guess the labels.
      Example ex_ = ex.copyData();
      // Cache the hypergraph
      Hypergraph Hq = modelA.createHypergraph(ex_, params.weights, counts.weights, 1.0/examples.size());
      Hq.computePosteriors(false);
      Hq.fetchPosteriors(false); // Places the posterior expectation $E_{Y|X}[\phi]$ into counts
      // Fill in the ex_.h values
      Hq.fetchBestHyperpath(ex_);

      for( int l = 0; l < ex.h.length; l++ )  
        labelMapping[ex.h[l]][ex_.h[l]] += 1; 
    }
    //if( debug )
      LogInfo.dbg( "Label mapping: \n" + Fmt.D( labelMapping ) );
    double acc = bestAccuracy( labelMapping );
    LogInfo.end_track();

    return acc;
  }

  /**
   * Uses method of moments to find moments along bottlenecks,
   * initializes parameters using them, and runs EM.
   */
  public ParamsVec solveBottleneckEM( List<Example> data ) {
    // Extract measurements via moments
    // Get moments
    ParamsVec bottleneckMeasurements = solveBottleneck( data );
    bottleneckMeasurements.write(Execution.getFile("measurements.counts"));

    ParamsVec initialParams = modelA.newParamsVec();
    ParamsVec beta = new ParamsVec( bottleneckMeasurements );

    initialParams.initRandom(genOpts.trueParamsRandom, genOpts.trueParamsNoise);
    beta.initRandom(genOpts.trueParamsRandom, genOpts.trueParamsNoise);

    // Use these measurements to solve for parameters
    ParamsVec theta_ = measurementsSolver.solveMeasurements(
        modelA, modelB, data, bottleneckMeasurements, initialParams, beta).getFirst();
    
    // solve EM
    return theta_;
  }

  public void setModel(Model model, int L) {
    this.modelA = model;
    // Run once to just instantiate features
    model.createHypergraph(L, null, null, 0);
  }

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
    @Option(gloss="Random seed for generating artificial data") public Random genRandom = new Random(1);
    @Option(gloss="Random seed for the true modelA") public Random trueParamsRandom = new Random(1);
    @Option(gloss="Number of examples to generate") public int genNumExamples = 100;
    @Option(gloss="How much variation in true parameters") public double trueParamsNoise = 1;
  }
  @OptionSet(name="gen") public GenerationOptions genOpts = new GenerationOptions();;

  /**
   * Generates random data from the modelA.
   *  - Uses genRand as a seed.
   */
  ParamsVec generateParameters( Model model, GenerationOptions opts ) {
    ParamsVec trueParams = model.newParamsVec();
    trueParams.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
    return trueParams;
  }

  /**
   * Generates random data from the modelA.
   *  - Uses genRand as a seed.
   */
  List<Example> generateData( Model model, ParamsVec params, GenerationOptions opts ) {
    LogInfo.begin_track("generateData");
    ParamsVec counts = model.newParamsVec();
    Hypergraph<Example> Hp = model.createHypergraph(modelOpts.L, params.weights, counts.weights, 1);
    // Necessary preprocessing before you can generate hyperpaths
    Hp.computePosteriors(false);
    Hp.fetchPosteriors(false);

    List<Example> examples = new ArrayList<Example>();

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
        MixtureModel model = new MixtureModel();
        model.L = opts.L;
        model.D = opts.D;
        this.modelA = model;
        this.modelB = model;
        break;
      }
      case hmm: {
        HiddenMarkovModel model = new HiddenMarkovModel();
        model.L = opts.L;
        model.D = opts.D;
        this.modelA = model;
        this.modelB = model;
        break;
      }
      case tallMixture: {
        TallMixture model = new TallMixture();
        model.L = opts.L;
        model.D = opts.D;
        this.modelA = model;
        throw new RuntimeException("Tall mixture not implemented");
//        this.modelB = model;

        //break;
      }
      case grid: {
        GridModel model = new GridModel(opts.L, opts.D);
        modelA = model;

        throw new RuntimeException("Tall mixture not implemented");
//        break;
      }
      default:
        throw new RuntimeException("Unhandled modelA type: " + opts.modelType);
    }
    modelA.K = opts.K;
    modelB.K = opts.K;

    modelA.createHypergraph(opts.L, null, null, 0);
    modelA.createHypergraph(opts.L, null, null, 0);
  }

  public void run() {

    // Setup modelA, modelB
    initializeModels( modelOpts );

    // Generate parameters
    ParamsVec trueParams = generateParameters( modelA, genOpts );
    // Get true parameters
    List<Example> data = generateData( modelA, trueParams, genOpts );

    analysis = new Analysis( modelA, trueParams, data );

    // Run the bottleneck spectral algorithm
    ParamsVec params = solveBottleneckEM(data);

    // Return the error in estimate
    analysis.reportParams( params );
  }

  public static void main(String[] args) {
    Execution.run(args, new SpectralMeasurements() );
  }
}


