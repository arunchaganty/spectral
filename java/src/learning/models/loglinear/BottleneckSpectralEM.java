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
  @Option(gloss="Type of optimization to use") public boolean useLBFGS = true;
  @Option(gloss="Use expected measurements (with respect to true distribution)") public boolean expectedMeasurements = true;
  @Option(gloss="Include each (true) measurement with this prob") public double measurementProb = 1;
  @Option(gloss="Initialize using measurements?") public boolean useMeasurements = true;
  @Option(gloss="Number of iterations before switching to full unsupervised") public int numMeasurementIters = Integer.MAX_VALUE;

  @Option(gloss="Smooth measurements") public double smoothMeasurements = 0.0;

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
      LogInfo.logsForce("countsError(%s)=%f [%s] - [%s]", Fmt.D(measuredFeatures), err, Fmt.D(estimatedCounts.weights), Fmt.D(trueCounts.weights));
      return err;
    }
    public double reportCounts(ParamsVec estimatedCounts) {
      boolean[] allMeasuredFeatures = new boolean[estimatedCounts.numFeatures];
      Arrays.fill( allMeasuredFeatures, true );

      return reportParams(estimatedCounts, allMeasuredFeatures);
    }
  }
  public Analysis analysis;

  Maximizer newMaximizer() {
    if (useLBFGS) return new LBFGSMaximizer(backtrack, lbfgs);
    return new GradientMaximizer(backtrack);
  }

  ////////////////////////////////////////////////////////
  
  // Algorithm parameters
  Model model;

  /**
   * Unrolls data along bottleneck nodes and uses method of moments
   * to return expected potentials
   */
  Pair<ParamsVec,boolean[]> solveBottleneck( final List<Example> data ) {
    LogInfo.begin_track("solveBottleneck");
    ParamsVec measurements = model.newParamsVec();
    boolean[] measuredFeatures = new boolean[measurements.numFeatures];

    // Construct data. For now, just return expected counts
    if (expectedMeasurements) {
      assert( analysis != null );
      // TODO: Revert to the original.
      Random random = new Random(3);
      //measuredFeatures[0] = true;
      for (int j = 0; j < model.numFeatures(); j++) {
        measuredFeatures[j] = random.nextDouble() < measurementProb;
        if (measuredFeatures[j])
          measurements.weights[j] = analysis.trueCounts.weights[j];
      }
    } else {
      //assert(false); // Don't do this yet.
      // Construct triples of three observed variables around the hidden
      // node.
      //
      int L, K, D;
      Iterator<double[][]> dataSeq;
      if( model instanceof MixtureModel) {
        final MixtureModel mixModel = (MixtureModel) model;
        K = mixModel.K; D = mixModel.D; L = mixModel.L;

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
        K = hmmModel.K; D = hmmModel.D; L = hmmModel.L;

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
      } else {
        throw new RuntimeException("Unhandled model type: " + model.getClass() );
      }

      TensorMethod algo = new TensorMethod();

      Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> 
        bottleneckMoments = algo.recoverParameters( K, D, dataSeq );

      double[] pi = MatrixFactory.toVector( bottleneckMoments.getValue0() );
      MatrixOps.projectOntoSimplex( pi, smoothMeasurements );
      SimpleMatrix M[] = {bottleneckMoments.getValue1(), bottleneckMoments.getValue2(), bottleneckMoments.getValue3()};

      // Set appropriate measuredFeatures to observed moments
      if( model instanceof MixtureModel ) {
        assert( M[2].numRows() == D );
        assert( M[2].numCols() == K );
        // Each column corresponds to a particular hidden moment.
        // Project onto the simplex
        M[2] = MatrixOps.projectOntoSimplex( M[2], smoothMeasurements );
        Execution.putOutput("moments.pi", MatrixFactory.fromVector(pi) );
        Execution.putOutput("moments.M3", M[2]);

        for( int h = 0; h < K; h++ ) {
          for( int d = 0; d < D; d++ ) {
            // Assuming identical distribution.
            int f = measurements.featureIndexer.getIndex(new UnaryFeature(h, "x="+d));
            measuredFeatures[f] = true;
            // multiplying by pi to go from E[x|h] -> E[x,h]
            // multiplying by 3 because true.counts aggregates
            // over x1, x2 and x3.
            measurements.weights[f] = L * M[2].get( d, h ) * pi[h]; 
          }
        }
        Execution.putOutput("moments.params", MatrixFactory.fromVector(measurements.weights));
      } else if( model instanceof HiddenMarkovModel ) {
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
        T = MatrixOps.projectOntoSimplex( T, smoothMeasurements );
        Execution.putOutput("moments.pi", Fmt.D(pi));
        Execution.putOutput("moments.O", O);
        Execution.putOutput("moments.T", T);

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
          if( true ) {
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
      } else {
        throw new RuntimeException("Unhandled model type: " + model.getClass() );
      }
    }
    LogInfo.end_track("solveBottleneck");

    if(analysis != null) analysis.reportCounts( measurements, measuredFeatures );

    return new Pair<>(measurements, measuredFeatures);
  }

  String logStat(String key, Object value) {
    LogInfo.logs("%s = %s", key, value);
    Execution.putOutput(key, value);
    return key+"="+value;
  }

  boolean optimize( Maximizer maximizer, Objective state, int numIters, String label ) {
    LogInfo.begin_track("optimize", label);
    state.invalidate();
    boolean done = false;
    // E-step
    int iter;

    PrintWriter out = IOUtils.openOutHard(Execution.getFile(label + ".events"));

    for (iter = 0; iter < numIters && !done; iter++) {
      LogInfo.begin_track("Iteration %s/%s", iter, numIters);
      done = maximizer.takeStep(state);
      LogInfo.logs("objective=%f", state.value());
      LogInfo.logs("counts=%s", Fmt.D(state.counts.weights));

      // Logging stuff
      List<String> items = new ArrayList<String>();
      int perm[] = new int[model.K];
      items.add("iter="+iter);
      items.add(logStat("paramsError", state.params.computeDiff(analysis.trueParams, perm)));
      items.add(logStat("paramsPerm", Fmt.D(perm)));
      items.add(logStat("countsError", state.counts.computeDiff(analysis.trueCounts, perm)));
      items.add(logStat("muError", state.mu.computeDiff(analysis.trueCounts, perm)));
      items.add(logStat("countsPerm", Fmt.D(perm)));
      items.add(logStat("eObjective", state.value()));
      out.println(StrUtils.join(items, "\t"));
      out.flush();

      // The EMObjective's per-example counts are stored in mu.
      // The global term's counts are stored in counts.
      analysis.reportCounts( state.mu );
      analysis.reportCounts( state.counts );

      LogInfo.end_track();
    }
    LogInfo.end_track();
    return done && iter == 1;  // Didn't make any updates
  }
  boolean optimize( Maximizer maximizer, Objective state, int numIters ) {
    return optimize( maximizer, state, numIters, "optimize" );
  }

  abstract class Objective implements Maximizer.FunctionState {
    public Model model;
    public double objective;
    public ParamsVec params;  // Canonical parameters
    public ParamsVec gradient;  // targetCounts - predCounts - \nabla regularization

    // Optimization state
    public boolean objectiveValid = false;
    public boolean gradientValid = false;

    public Hypergraph Hp;
    public ParamsVec counts;  
    public ParamsVec mu; // Expected counts; ether measurements or from hypergraph.

    public double regularization;

    public Objective(Model model, ParamsVec params, 
        double regularization) {
      this.model = model;
      this.params = params;
      this.gradient = model.newParamsVec();
      this.regularization = regularization;

      this.counts = model.newParamsVec();
      this.Hp = model.createHypergraph(null, params.weights, counts.weights, 1);
      this.mu = model.newParamsVec();
    }

    public void invalidate() { objectiveValid = gradientValid = false; }
    public double[] point() { return params.weights; }
    public double value() { compute(false); return objective; }
    public double[] gradient() { compute(true); return gradient.weights; }

    public abstract void compute(boolean needGradient);
  }

  class MomentMatchingObjective extends Objective {
    boolean[] measuredFeatures;

    // TODO: generalize to arbitrary quadratic (A w - b)^2

    public MomentMatchingObjective(Model model, ParamsVec params, double regularization, 
        ParamsVec measurements, boolean[] measuredFeatures, 
        Random initRandom, double initNoise) {
      super(model, params, regularization);

      this.mu = measurements;
      this.measuredFeatures = measuredFeatures;

      // Create state
      this.params.initRandom(initRandom, initNoise);
    }

    public void compute(boolean needGradient) {
      if (needGradient ? gradientValid : objectiveValid) return;
      objectiveValid = true;

      // Objective is \theta^T \tau - A(\theta) 
      objective = 0.0;
      // \theta^T \tau 
      objective += MatrixOps.dot( params.weights, mu.weights, measuredFeatures );

      // A(\theta)
      counts.clear(); Hp.computePosteriors(false);
      objective -= Hp.getLogZ();

      if (regularization > 0) {
        for (int j = 0; j < model.numFeatures(); j++) {
          if (!measuredFeatures[j]) continue;
          objective -= 0.5 * regularization * params.weights[j] * params.weights[j];
        }
      }
      // Compute objective value
      LogInfo.logs("objective: %f", objective);

      // Compute gradient (if needed)
      // \tau - E_{\theta}[\phi]
      if (needGradient) {
        gradientValid = true;
        gradient.clear();

        // Compute E_{\theta}[\phi]
        Hp.fetchPosteriors(false);

        logs("objective = %s, gradient (%s): [%s] - [%s] - %s [%s]", 
            Fmt.D(objective), Fmt.D(measuredFeatures), Fmt.D(mu.weights), Fmt.D(counts.weights), regularization, Fmt.D(params.weights));
        for (int j = 0; j < model.numFeatures(); j++) {
          if (!measuredFeatures[j]) continue;
          // takes \tau (from target) - E(\phi) (from pred).
          gradient.weights[j] += mu.weights[j] - counts.weights[j];

          // Regularization
          if (regularization > 0)
            gradient.weights[j] -= regularization * params.weights[j];
        }
        LogInfo.logs("gradient: %s", Fmt.D(gradient.weights));
      }
    }
  }

  /**
   * Potentially uses moments to find an initial set of parameters that
   * match it. 
   *  - Solved as a restricted M-step, minimizing $\theta^T \tau -
   *    A(\theta)$ or matching $E_\theta[\phi(x) = \tau$.
   */
  ParamsVec initializeParameters(final ParamsVec measurements, final boolean[] measuredFeatures) {
    LogInfo.begin_track("initialize parameters");
    // The objective function!
    ParamsVec params = model.newParamsVec();
    Maximizer maximizer = newMaximizer();
    MomentMatchingObjective state = new MomentMatchingObjective(
        model, params, eRegularization, 
        measurements, measuredFeatures,
        initRandom, initParamsNoise);
    
    // Optimize
    optimize( maximizer, state, 1000, "initialization" );
    LogInfo.end_track();
    return params;
  }
  ParamsVec initializeParameters(ParamsVec moments) {
      boolean[] allMeasuredFeatures = new boolean[moments.numFeatures];
      Arrays.fill( allMeasuredFeatures, true );
      return initializeParameters( moments, allMeasuredFeatures );
  }
  ParamsVec initializeParameters() {
    ParamsVec init = model.newParamsVec();
    init.initRandom( initRandom, initParamsNoise );
    return init;
  }

  class EMObjective extends Objective {
    List<Example> examples;

    public EMObjective(Model model, ParamsVec params, double regularization,
        List<Example> examples) {
      super(model, params, regularization);

      this.examples = examples;
    }

    void computeExpectedCounts() {
      mu.clear();
      for (Example ex : examples) {
        Hypergraph Hq = ex.Hq;
        // Cache the hypergraph
        if (Hq == null) Hq = model.createHypergraph(ex, params.weights, mu.weights, 1.0/examples.size());
        if (storeHypergraphs) ex.Hq = Hq;

        Hq.computePosteriors(false);
        Hq.fetchPosteriors(false); // Places the posterior expectation $E_{Y|X}[\phi]$ into counts
      }
      // At the end of this routine, 
      // mu contains $E_{Y|X}[\phi(X)]$ $\phi(x)$ are features.
    }

    public void compute(boolean needGradient) {
      // Always recompute for now.
      //if (needGradient ? gradientValid : objectiveValid) return;
      objectiveValid = true;

      computeExpectedCounts();

      // Objective is \theta^T \mu - A(\theta) 
      objective = 0.0;
      // \theta^T \tau 
      objective += MatrixOps.dot( params.weights, mu.weights );

      // A(\theta)
      counts.clear(); Hp.computePosteriors(false); Hp.fetchPosteriors(false);
      objective -= Hp.getLogZ();

      if (regularization > 0) {
        for (int j = 0; j < model.numFeatures(); j++) {
          objective -= 0.5 * regularization * params.weights[j] * params.weights[j];
        }
      }
      // Compute objective value
      LogInfo.logs("objective: %f", objective);

      // Compute gradient (if needed)
      // \mu - E_{\theta}[\phi]
      if (needGradient) {
        gradient.clear();
        gradientValid = true;

        logs("objective = %s, gradient: [%s] - [%s] - %s [%s]", 
            Fmt.D(objective), Fmt.D(mu.weights), Fmt.D(counts.weights), regularization, Fmt.D(params.weights));
        for (int j = 0; j < model.numFeatures(); j++) {
          // takes \tau (from target) - E(\phi) (from pred).
          gradient.weights[j] += mu.weights[j] - counts.weights[j];

          // Regularization
          if (regularization > 0)
            gradient.weights[j] -= regularization * params.weights[j];
        }
        LogInfo.logs("gradient: %s", Fmt.D(gradient.weights));
      }
    }
  }

  /**
   * Solves EM for the model using data and initial parameters.
   */
  ParamsVec solveEM(List<Example> data, ParamsVec initialParams) {
    LogInfo.begin_track("solveEM");
    Maximizer maximizer = newMaximizer();
    EMObjective state = new EMObjective(
        model, initialParams, mRegularization,
        data);
    // Optimize
    optimize( maximizer, state, numIters, "em" );
    state.params.write( Execution.getFile("params") );
    state.counts.write( Execution.getFile("counts") );
    LogInfo.end_track("solveEM");

    return initialParams;
  }

  /**
   * Uses method of moments to find moments along bottlenecks,
   * initializes parameters using them, and runs EM.
   */
  ParamsVec solveBottleneckEM( List<Example> data ) {
    // Extract measurements via moments
    ParamsVec initialParams;
    if( useMeasurements ) {
      // Get moments
      Pair<ParamsVec,boolean[]> bottleneckCounts = solveBottleneck( data );

      ParamsVec expectedCounts = bottleneckCounts.getFirst();
      boolean[] measuredFeatures = bottleneckCounts.getSecond();
      expectedCounts.write(Execution.getFile("measured_counts"));
      // initialize parameters from moments
      initialParams = initializeParameters( expectedCounts, measuredFeatures );
    } else { 
      initialParams = initializeParameters();
    }
    initialParams.write(Execution.getFile("params.init"));
    
    // solve EM
    return solveEM( data, initialParams );
  }

  void setModel(Model model) {
    this.model = model;
  }

  ///////////////////////////////////
  // Instantiation stuff

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
      //LogInfo.logs("x = %s", Fmt.D(ex.x));
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

  public void run() {

    // Setup; generate model 
    Model model = generateModel( modelOpts );
    setModel( model );

    // Generate parameters
    ParamsVec trueParams = generateParameters( model, genOpts );
    analysis = new Analysis( model, trueParams );

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


