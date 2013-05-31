/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.applications;

import learning.Misc;
import learning.linalg.*;
import learning.data.*;
import learning.em.*;
import learning.spectral.*;
import learning.models.HiddenMarkovModel;
import learning.models.HiddenMarkovModel.Params;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

import java.io.*;
import java.util.*;
import static learning.Misc.*;

/**
 * Perform POS induction, aka HMM learning.
 */
public class POSInduction implements Runnable {

  //@OptionSet(name="bottleneck")
  //public BottleneckSpectralEM algo = new BottleneckSpectralEM();
  @Option(gloss="Seed to generate initial point")
  public Random initRandom = new Random(1);
  @Option(gloss="Noise in parameters")
  public double initParamsNoise = 1.0;
  @Option(gloss="useLBFGS")
  public boolean useLBFGS = false;


  @Option(gloss="em iterations")
  public int iterations = 1000;
  @Option(gloss="em eps")
  public double eps = 1e-4;

  @Option(gloss="Measurements Path (pre-cached measurements only) - empty for no measurements")
  public String measurementsPath = "";
  @Option(gloss="smoothMeasurements")
  public double smoothMeasurements = 0.0;

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

  public double reportAccuracy( HiddenMarkovModel model, ParsedCorpus C ) {
    int K = C.getTagDimension();
    double[][] confusion = new double[K][K];
    for( int n = 0; n < C.getInstanceCount(); n++ ) {
      int[] l = C.L[n];
      int[] l_ = model.viterbi( C.C[n] );

      for( int i = 0; i < l.length; i++ )  
        confusion[l[i]][l_[i]] += 1; 
    }
    double acc = bestAccuracy( confusion);

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

    double lhood = Double.NEGATIVE_INFINITY;
    for( int iter = 0; iter < iterations; iter++ ) {
      double lhood_ = model.compute(params, C.C, params_);
      double diff = lhood_ - lhood;
      LogInfo.logs( "%f - %f = %f", lhood_, lhood, diff);
      assert( diff >= 0 );
      
      // Update with parameters.
      lhood = lhood_;
      // Copy params_ -> params
      System.arraycopy( params_, 0, params, 0, params_.length );
      model.params.updateFromVector( params );

      // Report
      List<String> items = new ArrayList<String>();
      items.add("iter="+iter);
      items.add(logStat("lhood", lhood));
      items.add(logStat("accuracy", reportAccuracy( model, C ) ) );
      eventsOut.println(StrUtils.join(items, "\t"));
      eventsOut.flush();

      if( diff < eps ) break;
    }

    LogInfo.end_track();

    return lhood;
  }

  @SuppressWarnings("unchecked")
  public Params loadMeasurements( String filename ) {
    try {
      // Open the file
      ObjectInputStream in = new ObjectInputStream( new FileInputStream( filename ) ); 
      Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix>
        measurements = (Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix>) in.readObject();
      in.close();

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
      return params;

    } catch( IOException e ) {
      LogInfo.fail(e);
    } catch( ClassNotFoundException e ) {
      LogInfo.fail(e);
    }
    return null;
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

      // Possibly load the measurements
      Params params;
      if( measurementsPath.trim().length() != 0 ) {
        // Get measurements from path.
        params = loadMeasurements( measurementsPath );
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


