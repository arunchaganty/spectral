/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.applications;

import fig.basic.*;
import fig.exec.Execution;
import learning.data.ComputableMoments;
import learning.data.Corpus;
import learning.data.MomentComputationWorkers;
import learning.data.ParsedCorpus;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import learning.models.HiddenMarkovModel;
import learning.models.HiddenMarkovModel.Params;
import learning.spectral.TensorMethod;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static fig.basic.LogInfo.*;

/**
 * Perform POS induction, aka HMM learning.
 */
public class POSInduction implements Runnable {

  //@OptionSet(name="bottleneck")
  //public BottleneckSpectralEM algo = new BottleneckSpectralEM();
  @Option(gloss="Seed to generate initial point")
  public Random initRandom = new Random();
  @Option(gloss="Noise in parameters")
  public double initParamsNoise = 1.0;
  @Option(gloss="useLBFGS")
  public boolean useLBFGS = false;

  @OptionSet(name="TensorMethod")
  public TensorMethod tensorMethod = new TensorMethod();

  @Option(gloss="em iterations")
  public int iterations = 200;
  @Option(gloss="em eps")
  public double eps = 1e-4;

  @Option(gloss="Initialize using spectral")
  public boolean initializeSpectral = false;
  @Option(gloss="smoothMeasurements")
  public double smoothMeasurements = 0.0;
  @Option(gloss="Number of threads to compute measurements")
  public int nThreads = 1;


  @Option(gloss="File containing word-index representation of data", required=true)
    public String dataPath;
  @Option(gloss="File containing word-index to word map", required=true)
    public String mapPath;
  @Option(gloss="File containing tag-index of labelled data")
    public String labelledDataPath = null;
  @Option(gloss="File containing tag-index to tag map")
    public String labelledMapPath = null;
  @Option(gloss="Possibly truncate the number of sentences read")
    public int maxSentences = Integer.MAX_VALUE;

  // public List<Example> corpusToExamples( Corpus C, int maxN ) {
  //   List<Example> examples = new ArrayList<Example>();

  //   for( int i = 0; i < C.getInstanceCount() && i < maxN; i++ ) {
  //     examples.add( new Example( C.C[i] ) );
  //   }

  //   return examples;
  // }
  // public List<Example> corpusToExamples( ParsedCorpus C, int maxN ) {
  //   List<Example> examples = new ArrayList<Example>();

  //   for( int i = 0; i < C.getInstanceCount() && i < maxN; i++ ) {
  //     Example ex = new Example( C.C[i], C.L[i] );
  //     examples.add( ex );
  //     // LogInfo.logs( Fmt.D( ex.x ) );
  //   }
  //   return examples;
  // }
  public void truncate( ParsedCorpus C, int maxN ) {
    if( maxN > C.getInstanceCount() ) return;

    int[][] C_ = new int[ maxN ][];
    int[][] L_ = new int[ maxN ][];
    for( int i = 0; i < maxN; i++ ) {
      C_[i] = new int[ C.C[i].length ];
      L_[i] = new int[ C.L[i].length ];
      System.arraycopy( C.C[i], 0, C_[i], 0, C.C[i].length );
      System.arraycopy( C.L[i], 0, L_[i], 0, C.L[i].length );
    }
    C.C = C_;
    C.L = L_;
  }

  public void printTopK(HiddenMarkovModel model, ParsedCorpus C, int[] perm) {
    int K = model.getStateCount();
    // (b) Print top 10 words
    LogInfo.begin_track("Top-k words");
    int TOP_K = 20;
    for( int k = 0; k < K; k++ ) {
      StringBuilder topk = new StringBuilder();
      topk.append(C.tagDict[k]).append(": ");

      int k_ = perm[k];
      Integer[] sortedWords = MatrixOps.argsort(MatrixOps.col(model.getO(), k_));
      for(int i = 0; i < TOP_K; i++ ) {
        topk.append(C.dict[sortedWords[i]]).append(", ");
      }
      logsForce(topk);
    }
    LogInfo.end_track("Top-k words");
  }


  public double reportAccuracy( HiddenMarkovModel model, ParsedCorpus C ) {
    begin_track("best-match accuracy");
    // Create a confusion matrix
    int K = C.getTagDimension();
    double[][] confusion = new double[K][K];
    for( int n = 0; n < C.getInstanceCount(); n++ ) {
      int[] l = C.L[n];
      int[] l_ = model.viterbi( C.C[n] );

      for( int i = 0; i < l.length; i++ )  
        confusion[l[i]][l_[i]] += 1; 
    }

    //double acc = bestAccuracy(confusion);
    // Additional debug information

    // Find best alignment
    BipartiteMatcher matcher = new BipartiteMatcher();
    int[] perm = matcher.findMaxWeightAssignment(confusion);
    // Compute hamming score
    long correct = 0;
    long total = 0;
    for( int k = 0; k < K; k++ ) {
      for( int k_ = 0; k_ < K; k_++ ) {
        total += confusion[k][k_];
      }
      correct += confusion[k][perm[k]];
    }
    double acc = (double) correct/ (double) total;

    // Now (a) print a confusion matrix
    LogInfo.begin_track("Confusion matrix");
    StringBuilder table = new StringBuilder();
    table.append( "\t" );
    for( int k = 0; k < K; k++ ) {
      table.append(C.tagDict[k]).append("\t");
    }
    table.append("\n");
    for( int k = 0; k < K; k++ ) {
      table.append(C.tagDict[k]).append("\t");
      for( int k_ = 0; k_ < K; k_++ ) {
        table.append(confusion[k][perm[k_]]).append("\t");
      }
      table.append( "\n" );
    }
    logsForce(table);
    LogInfo.end_track("Confusion matrix");

    // (b) Print top 10 words
    printTopK(model, C, perm);
    end_track("best-match accuracy");

    return acc;
  }

  /**
   * Greedy matching accuracy
   */
  public double reportVsAllAccuracy( HiddenMarkovModel model, ParsedCorpus C ) {
    begin_track("vs-all accuracy");
    // Create a confusion matrix
    int K = C.getTagDimension();
    double[][] confusion = new double[K][K];
    for( int n = 0; n < C.getInstanceCount(); n++ ) {
      int[] l = C.L[n];
      int[] l_ = model.viterbi( C.C[n] );

      for( int i = 0; i < l.length; i++ )
        // NOTE: This is different than the alignment in reportAccuracy
        confusion[l_[i]][l[i]] += 1;
    }

    // Find greedy alignment
    int[] perm = new int[K];
    for(int k = 0; k < K; k++) perm[k] = MatrixOps.argmax(confusion[k]);

    // Compute hamming score
    long correct = 0;
    long total = 0;
    for( int k = 0; k < K; k++ ) {
      for( int k_ = 0; k_ < K; k_++ ) {
        total += confusion[k][k_];
      }
      correct += confusion[k][perm[k]];
    }
    double acc = (double) correct/ (double) total;

    // Now (a) print a confusion matrix
    LogInfo.begin_track("Confusion matrix");
    StringBuilder table = new StringBuilder();
    table.append( "\t" );
    for( int k = 0; k < K; k++ ) {
      table.append(C.tagDict[k]).append("\t");
    }
    table.append( "\n" );
    for( int k = 0; k < K; k++ ) {
      table.append(C.tagDict[k]).append("\t");
      for( int k_ = 0; k_ < K; k_++ ) {
        table.append(confusion[k][perm[k_]]).append("\t");
      }
      table.append( "\n" );
    }
    logsForce(table);
    LogInfo.end_track("Confusion matrix");

    // (b) Print top 10 words
    LogInfo.begin_track("Top-k words");
    int TOP_K = 20;
    for( int k = 0; k < K; k++ ) {
      StringBuilder topk = new StringBuilder();
      topk.append(C.tagDict[k]).append(": ");

      int k_ = perm[k];
      Integer[] sortedWords = MatrixOps.argsort(MatrixOps.col(model.getO(), k_));
      for(int i = 0; i < TOP_K; i++ ) {
        topk.append(C.dict[sortedWords[i]]).append(", ");
      }
      logsForce(topk);
    }
    LogInfo.end_track("Top-k words");

    end_track("vs-all accuracy");
    return acc;
  }

  String logStat(String key, Object value) {
    LogInfo.logs("%s = %s", key, value);
    Execution.putOutput(key, value);
    return key+"="+value;
  }

  // WARNING: I'm updating the model with params.
  public double optimize( HiddenMarkovModel model, double[] params, ParsedCorpus C ) {
    assert( params.length == model.numFeatures() );
    LogInfo.begin_track( "em.optimize" );

    PrintWriter eventsOut = IOUtils.openOutHard(Execution.getFile("events"));

    double[] params_ = new double[params.length];

    // UGH.
    model.params.updateFromVector( params );
    {
      int[] perm = new int[C.getTagDimension()];
      for(int i = 0; i < C.getTagDimension(); i++) perm[i] = i;
      printTopK(model, C, perm);
    }

    double lhood = Double.NEGATIVE_INFINITY;
    for( int iter = 0; iter < iterations; iter++ ) {
      double lhood_ = model.compute(C.C, params_);
      double diff = lhood_ - lhood;
      LogInfo.logs( "%f - %f = %f", lhood_, lhood, diff);
      //assert( diff >= -1 );
      
      // Update with parameters.
      lhood = lhood_;
      // Copy params_ -> params
      System.arraycopy( params_, 0, params, 0, params_.length );
      model.params.updateFromVector( params );

      // Report
      List<String> items = new ArrayList<>();
      items.add(logStat("iter", iter+1));
      items.add(logStat("lhood", lhood));
      items.add(logStat("accuracy", reportAccuracy( model, C ) ) );
      items.add(logStat("all-accuracy", reportVsAllAccuracy( model, C ) ) );
      eventsOut.println(StrUtils.join(items, "\t"));
      eventsOut.flush();

      //if( Math.abs(diff) < eps ) break;
    }

    LogInfo.end_track();

    return lhood;
  }

  public ComputableMoments corpusToMoments(final Corpus C) {
    return new ComputableMoments() {
      @Override
      public MatrixOps.Matrixable computeP13() {
        return MomentComputationWorkers.matrixable(C, 0, 2, nThreads);
      }

      @Override
      public MatrixOps.Matrixable computeP12() {
        return MomentComputationWorkers.matrixable(C, 0, 1, nThreads);
      }

      @Override
      public MatrixOps.Matrixable computeP32() {
        return MomentComputationWorkers.matrixable(C, 2, 1, nThreads);
      }

      @Override
      public MatrixOps.Tensorable computeP123() {
        return MomentComputationWorkers.tensorable(C, 0, 1, 2, nThreads);
      }
    };
  }

  /**
   * Use tensor method to recover the parameters of the HMM.
   * @param C - Corpus
   * @return - Initial parameters
   */
  public Params spectralRecovery(Corpus C, int K) {
    LogInfo.begin_track("spectral-recovery");
    // Compute moments
    ComputableMoments moments = corpusToMoments(C);
    // Run tensor recovery
    Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> measurements =
        tensorMethod.randomizedRecoverParameters(K, moments);

    // Populate params
    // construct O T and pi
    SimpleMatrix O = measurements.getValue2();
    SimpleMatrix T = O.pseudoInverse().mult( measurements.getValue3() );

    // Initialize pi to be random.
    SimpleMatrix pi = measurements.getValue0(); // This is just a random guess.

    // project and smooth
    // projectOntoSimplex normalizes columns!
    pi = MatrixOps.projectOntoSimplex( pi.transpose(), smoothMeasurements );
    T = MatrixOps.projectOntoSimplex( T, smoothMeasurements ).transpose();
    O = MatrixOps.projectOntoSimplex( O, smoothMeasurements ).transpose();

    Params params = new Params(
        MatrixFactory.toVector(pi),
        MatrixFactory.toArray(T),
        MatrixFactory.toArray(O));
    LogInfo.end_track("spectral-recovery");

    return params;
  }
  public Params spectralRecovery(ParsedCorpus C) {
    return spectralRecovery(C, C.getTagDimension());
  }

  public void run() {
    try { 
      LogInfo.begin_track("file-input");
      // Read data as word-index sequences
      ParsedCorpus C = ParsedCorpus.parseText( 
          dataPath, mapPath, 
          labelledDataPath, labelledMapPath );
      // Convert to Example sequences
      LogInfo.logs( "Corpus has %d instances, with %d words and %d tags",
          C.getInstanceCount(), C.getDimension(), C.getTagDimension() );
      truncate( C, maxSentences );
      LogInfo.end_track();
      ///
      
      HiddenMarkovModel model = new HiddenMarkovModel( 
          C.getTagDimension(), // K
          C.getDimension() ); // D
      LogInfo.end_track("file-input");

      // Possibly load the measurements
      Params params;
      if(initializeSpectral) {
        // Get measurements from path.
        params = spectralRecovery(C);
      } else {
        // random
        params = Params.uniformWithNoise( 
            initRandom,
            model.params.stateCount,
            model.params.emissionCount,
            initParamsNoise
            );
      }
      optimize( model, params.toVector(), C );

    } catch( IOException e ) {
      LogInfo.logsForce( e );
    }
  }
  
  public static void main(String[] args) {
    Execution.run(args, new POSInduction() );
  }
}


