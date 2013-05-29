/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.applications;

import learning.Misc;
import learning.linalg.*;
import learning.data.*;
import learning.spectral.TensorMethod;
import learning.models.loglinear.*;
import learning.models.loglinear.Models.HiddenMarkovModel;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

import java.io.*;
import java.util.*;

/**
 * Perform POS induction, aka HMM learning.
 */
public class POSInduction implements Runnable {

  @OptionSet(name="bottleneck")
  public BottleneckSpectralEM algo = new BottleneckSpectralEM();

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

  public List<Example> corpusToExamples( Corpus C, int maxN ) {
    List<Example> examples = new ArrayList<Example>();

    for( int i = 0; i < C.getInstanceCount() && i < maxN; i++ ) {
      examples.add( new Example( C.C[i] ) );
    }

    return examples;
  }
  public List<Example> corpusToExamples( ParsedCorpus C, int maxN ) {
    List<Example> examples = new ArrayList<Example>();

    for( int i = 0; i < C.getInstanceCount() && i < maxN; i++ ) {
      Example ex = new Example( C.C[i], C.L[i] );
      examples.add( ex );
      // LogInfo.logs( Fmt.D( ex.x ) );
    }
    return examples;
  }
  
  public void run() {
    try { 
      LogInfo.begin_track("file-input");
      // Read data as word-index sequences
      ParsedCorpus C = ParsedCorpus.parseText( 
          dataPath, mapPath, 
          labelledDataPath, labelledMapPath );
      // Convert to Example sequences
      List<Example> data = corpusToExamples( C, maxSentences );
      LogInfo.logs( "Corpus has %d instances, with %d words and %d tags",
          C.getInstanceCount(), C.getDimension(), C.getTagDimension() );
      LogInfo.end_track();

      // Run BottleneckSpectralEM
      HiddenMarkovModel model = new HiddenMarkovModel(
          C.getTagDimension(), // K
          C.getDimension(),  // D
          3 ); // L
      algo.setModel( model );
      ParamsVec params = algo.solveBottleneckEM( data );

      LogInfo.logs( Fmt.D(params.weights) );

    } catch( IOException e ) {
      LogInfo.logsForce( e );
    }
  }
  
  public static void main(String[] args) {
    Execution.run(args, new POSInduction() );
  }
}


