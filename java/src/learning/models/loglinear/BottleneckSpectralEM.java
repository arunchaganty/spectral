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

import static learning.models.loglinear.Models.*;
import static learning.Misc.*;

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

  @Option(gloss="Use cached bottleneck moments") public String cachedBottleneckMoments = null;

  @Option(gloss="Smooth measurements") public double smoothMeasurements = 0.0;
  @Option(gloss="Use T in BottleneckSpectralEM?") public boolean useTransitions = true;

  @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
  @OptionSet(name="backtrack") public BacktrackingLineSearch.Options backtrack = new BacktrackingLineSearch.Options();
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
     */
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

  Maximizer newMaximizer() {
    if (useLBFGS) return new LBFGSMaximizer(backtrack, lbfgs);
    return new GradientMaximizer(backtrack);
  }

  ////////////////////////////////////////////////////////

  String logStat(String key, Object value) {
    LogInfo.logs("%s = %s", key, value);
    Execution.putOutput(key, value);
    return key+"="+value;
  }

  ////////////////////////////////////////////////////////
  
  // Algorithm parameters
  Model model;

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
      throw new RuntimeException("Unhandled model type: " + model.getClass() );
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
      int L = ((MixtureModel)(model)).L; // Extract from the model in a clever way.
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
      int L = ((HiddenMarkovModel)(model)).L; // Extract from the model in a clever way.
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
      int L = ((GridModel)(model)).L; // Extract from the model in a clever way.
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
      throw new RuntimeException("Unhandled model type: " + model.getClass() );
    }
  
    LogInfo.end_track();
  }
  /**
   * Unrolls data along bottleneck nodes and uses method of moments
   * to return expected potentials
   */
  @SuppressWarnings("unchecked")
  Pair<ParamsVec,boolean[]> solveBottleneck( final List<Example> data ) {
    LogInfo.begin_track("solveBottleneck");
    ParamsVec measurements = model.newParamsVec();
    boolean[] measuredFeatures = new boolean[measurements.numFeatures];

    // Construct data. For now, just return expected counts
    if (expectedMeasurements) {
      assert( analysis != null );
      Random random = new Random(3);
      //measuredFeatures[0] = true;
      for (int j = 0; j < model.numFeatures(); j++) {
        measuredFeatures[j] = random.nextDouble() < measurementProb;
        if (measuredFeatures[j])
          measurements.weights[j] = analysis.trueCounts.weights[j];
      }
    } else {
      // Construct triples of three observed variables around the hidden
      // node.
      int K = model.K; int D = model.D;

      Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> bottleneckMoments = null;
      Iterator<double[][]> dataSeq = constructDataSequence( model, data );
      bottleneckMoments = algo.recoverParameters( K, D, dataSeq );
      populateFeatures( model, bottleneckMoments, measurements, measuredFeatures ); 
    }
    LogInfo.logs("sum_counts: " + MatrixOps.sum(measurements.weights));
    if(analysis != null)  LogInfo.logs("sum_counts (true): " + MatrixOps.sum(analysis.trueCounts.weights));
    if(analysis != null) analysis.reportCounts( measurements, measuredFeatures );

    LogInfo.end_track("solveBottleneck");

    return new Pair<>(measurements, measuredFeatures);
  }

  /////////////////////

  boolean optimize( Maximizer maximizer, Objective state, int numIters, List<Example> examples, String label ) {
    LogInfo.begin_track("optimize", label);
    state.invalidate();
    boolean done = false;
    // E-step
    int iter;

    PrintWriter out = IOUtils.openOutHard(Execution.getFile(label + ".events"));

    for (iter = 0; iter < numIters && !done; iter++) {
      LogInfo.begin_track("Iteration %s/%s", iter, numIters);
      done = maximizer.takeStep(state);
      state.performGradientCheck();
      LogInfo.logs("objective=%f", state.value());
      //LogInfo.logs("counts=%s", Fmt.D(state.counts.weights));

      // Logging stuff
      List<String> items = new ArrayList<String>();
      int perm[] = new int[model.K];
      items.add("iter="+iter);
      if(analysis != null) {
        items.add(logStat("paramsError", state.params.computeDiff(analysis.trueParams, perm)));
        items.add(logStat("paramsPerm", Fmt.D(perm)));
        items.add(logStat("countsError", state.getCounts().computeDiff(analysis.trueCounts, perm)));
        items.add(logStat("countsPerm", Fmt.D(perm)));
      }
      items.add(logStat("eObjective", state.value()));
      if( examples != null )
        items.add(logStat("accuracy", reportAccuracy( state.params, examples ) ) ); 
      out.println(StrUtils.join(items, "\t"));
      out.flush();

      // The EMObjective's per-example counts are stored in mu.
      // The global term's counts are stored in counts.
      if(analysis != null ) {
        analysis.reportCounts( state.getCounts() );
        //analysis.reportCounts( state.counts );
      }

      LogInfo.end_track();
    }
    LogInfo.end_track();
    return done && iter == 1;  // Didn't make any updates
  }

  // A term in the objective function.
  abstract class ObjectiveTerm {
    double value;
    ParamsVec gradient;

    // Populate |value| and |counts| based on the current state.
    public abstract void infer(boolean needGradient);
  }

  class GlobalTerm extends ObjectiveTerm {
    Model model;
    ParamsVec params; // A reference.
    Map<Hypergraph<Example>, Integer> Hp; // We maintain a different hypergraph for each length.

    public GlobalTerm(Model model, ParamsVec params, List<Example> examples) {
      this.model = model;
      this.params = params;
      this.gradient = model.newParamsVec();

      // Construct a H for each length in the examples.
      // HACK: Likely only to work for the examples we're using.
      Map<Integer, Integer> countProfile = getCountProfile(examples);
      LogInfo.begin_track("Constructing globalTerm");
      LogInfo.logs("Found %d unique lengths",  countProfile.size() );
      Hp = new HashMap<>();
      for( Map.Entry<Integer,Integer> pair : countProfile.entrySet() ) {
        int L = pair.getKey(); int cnt = pair.getValue();

        LogInfo.logs( "%d instances of %d length", cnt, L );
        Hp.put( model.createHypergraph(L, params.weights, gradient.weights, (double) cnt/examples.size()),
            cnt );
        //Hp.put( model.createHypergraph(L, params.weights, gradient.weights, (double) 1), 1 );
        //break; // Cheating because we're running out of memory.
      }
      LogInfo.end_track();
    }

    public void infer(boolean needGradient) {
      if (needGradient) gradient.clear();
      value = 0.0;
      int totalCount = 0;
      for( Hypergraph<Example> H : Hp.keySet() ) {
        int cnt = Hp.get(H);
        H.computePosteriors(false);
        if (needGradient) H.fetchPosteriors(false);
        value += (double) cnt * (H.getLogZ() - value)/(totalCount+cnt);
        totalCount += cnt;
      }
    }
  }

  // Linear function with given coefficients
  class LinearTerm extends ObjectiveTerm {
    ParamsVec params;
    ParamsVec coeffs; 
    boolean[] measuredFeatures;  // Which features to include

    public LinearTerm(ParamsVec params, ParamsVec coeffs, boolean[] measuredFeatures) {
      this.params = params;
      this.coeffs = coeffs;
      this.measuredFeatures = measuredFeatures;

      this.gradient = coeffs;
    }

    /**
     * Gives the objective + gradient.
     * L = <params,coeffs>
     * dL = coeffs
     */
    public void infer(boolean needGradient) {
      value = MatrixOps.dot( params.weights, coeffs.weights, measuredFeatures );
      if (needGradient) this.gradient = coeffs;
    }
  }

  class ExamplesTerm extends ObjectiveTerm {
    Model model;
    ParamsVec params;
    List<Example> examples;
    boolean storeHypergraphs;

    ExamplesTerm(Model model, ParamsVec params, List<Example> examples, boolean storeHypergraphs) {
      this.model = model;
      this.params = params;
      this.gradient = model.newParamsVec();
      this.examples = examples;
      this.storeHypergraphs = storeHypergraphs;
    }

    /**
     * Gives the objective + gradient.
     * L = log partition (A_i, B_i)
     * dL = expected counts (\mu, \tau)
     */
    public void infer(boolean needGradient) {
      logs("ExamplesTerm.infer");
      value = 0;
      if (needGradient) gradient.clear();
      for (Example ex : examples) {
        Hypergraph Hq = ex.Hq;
        // Cache the hypergraph
        if (Hq == null) Hq = model.createHypergraph(ex, params.weights, gradient.weights, 1.0/examples.size());
        if (storeHypergraphs) ex.Hq = Hq;

        Hq.computePosteriors(false);
        if (needGradient) Hq.fetchPosteriors(false); // Places the posterior expectation $E_{Y|X}[\phi]$ into counts
        value += Hq.getLogZ() * 1.0/examples.size();
      }
      // At the end of this routine, 
      // counts contains $E_{Y|X}[\phi(X)]$ $\phi(x)$ are features.
    }
  }

  class ExpectedLinearTerm extends ObjectiveTerm {
    Model model;
    ParamsVec params;
    List<Example> examples;
    boolean storeHypergraphs;

    ExpectedLinearTerm(Model model, ParamsVec params, List<Example> examples, boolean storeHypergraphs) {
      this.model = model;
      this.params = params;
      this.gradient = model.newParamsVec();
      this.examples = examples;
      this.storeHypergraphs = storeHypergraphs;
    }

    /**
     * Gives the objective + gradient.
     * L = log partition (A_i, B_i)
     * dL = expected counts (\mu, \tau)
     */
    public void infer(boolean needGradient) {
      logs("ExpectedExamplesTerm.infer");
      value = 0;
      gradient.clear();
      for (Example ex : examples) {
        Hypergraph Hq = ex.Hq;
        // Cache the hypergraph
        if (Hq == null) Hq = model.createHypergraph(ex, params.weights, gradient.weights, 1.0/examples.size());
        if (storeHypergraphs) ex.Hq = Hq;

        Hq.computePosteriors(false);
        Hq.fetchPosteriors(false); // Places the posterior expectation $E_{Y|X}[\phi]$ into counts
        //value += Hq.getLogZ() * 1.0/examples.size();
      }
      value = MatrixOps.dot(params.weights, gradient.weights);
      // At the end of this routine, 
      // counts contains $E_{Y|X}[\phi(X)]$ $\phi(x)$ are features.
    }
  }

  public static boolean[] allMeasuredFeatures(ParamsVec features) {
    boolean[] measuredFeatures = new boolean[features.numFeatures];
    Arrays.fill( measuredFeatures, true );
    return measuredFeatures;
  }

  // Implements an objective of target - pred - regularization
  abstract class Objective implements Maximizer.FunctionState {
    public Model model;
    public ParamsVec params;  // Canonical parameters that we are optimizing 

    public double objective; // Objective value
    public ParamsVec gradient;  // targetCounts - predCounts - \nabla regularization

    // Optimization state
    public boolean objectiveValid = false;
    public boolean gradientValid = false;

    ObjectiveTerm target; 
    ObjectiveTerm pred;
    boolean[] measuredFeatures;
    double regularization;

    protected Objective(Model model, ParamsVec params, 
        ObjectiveTerm target, ObjectiveTerm pred,
        boolean[] measuredFeatures,
        double regularization) {
      this.model = model;
      this.params = params;

      this.objective = 0;
      this.gradient = model.newParamsVec();

      this.target = target;
      this.pred = pred;
      this.measuredFeatures = measuredFeatures;
      this.regularization = regularization;
    }
    public void initRandom(Random initRandom, double initNoise) {
      // Create state
      this.params.initRandom(initRandom, initNoise);
    }

    public void invalidate() { objectiveValid = gradientValid = false; }
    public double[] point() { return params.weights; }
    public double value() { compute(false); return objective; }
    public double[] gradient() { compute(true); return gradient.weights; }

    public void compute(boolean needGradient) {
      if (needGradient ? gradientValid : objectiveValid) return;
      objectiveValid = true;

      target.infer(needGradient);
      pred.infer(needGradient);

      // Objective is \theta^T \tau - A(\theta) 
      objective = 0.0;
      objective = target.value - pred.value;
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
        gradient.clear();
        gradientValid = true;
        // Compute E_{\theta}[\phi]

        // logs("objective = %s, gradient (%s): [%s] - [%s] - %s [%s]", 
        //     Fmt.D(objective), Fmt.D(measuredFeatures), Fmt.D(mu.weights), Fmt.D(counts.weights), regularization, Fmt.D(params.weights));
        for (int j = 0; j < model.numFeatures(); j++) {
          if (!measuredFeatures[j]) continue;
          // takes \tau (from target) - E(\phi) (from pred).
          gradient.weights[j] += target.gradient.weights[j] - pred.gradient.weights[j];

          // Regularization
          if (regularization > 0)
            gradient.weights[j] -= regularization * params.weights[j];
        }
        // LogInfo.logs("gradient: %s", Fmt.D(gradient.weights));
      }
    }

    public void performGradientCheck() {
      double epsilon = 1e-4;
      // Save point
      double[] currentGradient = gradient();
      double[] currentPoint = point();

      // Set point to be +/- gradient
      for( int i = 0; i < currentPoint.length; i++ ) {
        params.weights[i] = currentPoint[i] + epsilon * currentGradient[i];
        double valuePlus = value();
        params.weights[i] = currentPoint[i] - epsilon * currentGradient[i];
        double valueMinus = value();
        params.weights[i] = currentPoint[i];

        double expectedValue = (valuePlus - valueMinus)/(2*epsilon);
        double actualValue = currentGradient[i];

        assert MatrixOps.equal( expectedValue, actualValue );
      }
    }

    abstract public ParamsVec getCounts();
  }

  /**
   * Create the MomentMatchingObjective. 
   * measurementsTerm - globalTerm - regularizer
   */
  class MomentMatchingObjective extends Objective {
    public MomentMatchingObjective(Model model, ParamsVec params, 
        List<Example> examples,
        ParamsVec measurements, boolean[] measuredFeatures, 
        double regularization) { 
      super(model, params, 
          new LinearTerm(params, measurements, measuredFeatures), // measurementsTerm
          new GlobalTerm(model, params, examples), // globalTerm
          measuredFeatures,
          regularization
          );
    }
    public ParamsVec getCounts() {
      return pred.gradient;
    }
  }

  class EMObjective extends Objective {
    List<Example> examples;

    public EMObjective(Model model, ParamsVec params, 
        List<Example> examples, boolean storeHypergraphs,
        double regularization) {
      super(model, params, 
          new ExpectedLinearTerm(model, params, examples, storeHypergraphs), // measurementsTerm
          new GlobalTerm(model, params, examples), // globalTerm
          allMeasuredFeatures(params),
          regularization);
    }
    public ParamsVec getCounts() {
      return target.gradient;
    }
  }

  /**
   * Uses the params and model to label the hidden labels for each
   * data point and reports Hamming errors. 
   */
  @SuppressWarnings("unchecked")
  public double reportAccuracy( ParamsVec params, List<Example> examples ) {
    LogInfo.begin_track("report-accuracy");
    ParamsVec counts = model.newParamsVec();
    int K = model.K;

    // Create a confusion matrix with prediced vs estimated label choices
    double[][] labelMapping = new double[K][K];
    for(int i = 0; i < examples.size(); i++ ) {
      Example ex = examples.get(i);
      // Create a copy of the example with just the data; use this to
      // guess the labels.
      Example ex_ = ex.copyData();
      // Cache the hypergraph
      Hypergraph Hq = model.createHypergraph(ex_, params.weights, counts.weights, 1.0/examples.size());
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
   * Potentially uses moments to find an initial set of parameters that
   * match it. 
   *  - Solved as a restricted M-step, minimizing $\theta^T \tau -
   *    A(\theta)$ or matching $E_\theta[\phi(x) = \tau$.
   */
  ParamsVec initializeParameters(final List<Example> examples, final ParamsVec measurements, final boolean[] measuredFeatures) {
    LogInfo.begin_track("initialize parameters");
    // The objective function!
    ParamsVec params = model.newParamsVec();
    Maximizer maximizer = newMaximizer();
    MomentMatchingObjective state = new MomentMatchingObjective(
        model, params, 
        examples,
        measurements, measuredFeatures,
        eRegularization); 
    state.initRandom(initRandom, initParamsNoise);
    
    // Optimize
    optimize( maximizer, state, eNumIters, examples, "initialization" );
    LogInfo.end_track();
    return params;
  }
  ParamsVec initializeParameters(final List<Example> examples, ParamsVec moments) {
      boolean[] allMeasuredFeatures = new boolean[moments.numFeatures];
      Arrays.fill( allMeasuredFeatures, true );
      return initializeParameters( examples, moments, allMeasuredFeatures );
  }
  ParamsVec initializeParameters() {
    ParamsVec init = model.newParamsVec();
    init.initRandom( initRandom, initParamsNoise );
    return init;
  }

  /**
   * Solves EM for the model using data and initial parameters.
   */
  ParamsVec solveEM(List<Example> data, ParamsVec initialParams) {
    LogInfo.begin_track("solveEM");
    LogInfo.logs( "Solving for %d parameters, using %d instances", 
        initialParams.numFeatures, data.size() );
    Maximizer maximizer = newMaximizer();
    EMObjective state = new EMObjective(
        model, initialParams, 
        data, storeHypergraphs,
        mRegularization);
    // Optimize
    optimize( maximizer, state, numIters, data, "em" );
    state.params.write( Execution.getFile("params") );
    state.getCounts().write( Execution.getFile("counts") );
    LogInfo.end_track("solveEM");

    return initialParams;
  }

  /**
   * Uses method of moments to find moments along bottlenecks,
   * initializes parameters using them, and runs EM.
   */
  public ParamsVec solveBottleneckEM( List<Example> data ) {
    // Extract measurements via moments
    ParamsVec initialParams;
    if( useMeasurements ) {
      // Get moments
      Pair<ParamsVec,boolean[]> bottleneckCounts = solveBottleneck( data );

      ParamsVec expectedCounts = bottleneckCounts.getFirst();
      boolean[] measuredFeatures = bottleneckCounts.getSecond();
      expectedCounts.write(Execution.getFile("measured_counts"));
      // initialize parameters from moments
      initialParams = initializeParameters( data, expectedCounts, measuredFeatures );
    } else { 
      initialParams = initializeParameters();
    }
    initialParams.write(Execution.getFile("params.init"));
    
    // solve EM
    return solveEM( data, initialParams );
  }

  public void setModel(Model model, int L) {
    this.model = model;
    // Run once to just instantiate features
    model.createHypergraph(L, null, null, 0);
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
        GridModel model = new GridModel(opts.L, opts.D);
        model_ = model;
        break;
      }
      default:
        throw new RuntimeException("Unhandled model type: " + opts.modelType);
    }

    model_.K = opts.K;
    return model_;
  }

  public void run() {

    // Setup; generate model 
    Model model = generateModel( modelOpts );
    setModel( model, modelOpts.L );

    // Generate parameters
    ParamsVec trueParams = generateParameters( model, genOpts );
    // Get true parameters
    List<Example> data = generateData( model, trueParams, genOpts );

    analysis = new Analysis( model, trueParams, data );

    // Run the bottleneck spectral algorithm
    ParamsVec params = solveBottleneckEM(data);

    // Return the error in estimate
    analysis.reportParams( params );
  }

  public static void main(String[] args) {
    Execution.run(args, new BottleneckSpectralEM() );
  }
}


