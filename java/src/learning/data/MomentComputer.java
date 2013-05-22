/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

import learning.data.Corpus;
import learning.data.ProjectedCorpus;
import learning.linalg.*;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.exec.Execution;

import java.io.*;

import org.ejml.simple.SimpleMatrix;

import org.javatuples.*;

/**
 * Efficiently computes moments of text corpora.
 */
public class MomentComputer implements Runnable {

  @Option(gloss="Number of threads to run for")
  public int nThreads;

  public MomentComputer( int nThreads ) {
    this.nThreads = nThreads;
  }
  public MomentComputer() {
    this( 4 );
  }

  /**
   * Compute moments for a range in the dataset.
   */
  protected class Worker extends Thread {
    public double[][] P12;
    public double[][] P13;
    public double[][] P23;
    public double[][][] P123;

    final protected RealSequenceData seqData; // Needs a copy to maintain a unique cache

    public int offset;
    public int length;
    public Worker( RealSequenceData seqData, int offset, int length ) {
      this.seqData = seqData;
      this.offset = offset;
      if( offset + length > seqData.getInstanceCount() )
        length = seqData.getInstanceCount() - offset;
      this.length = length;
    }

    public void run() {
      int D = seqData.getDimension();
      P12 = new double[D][D];
      P13 = new double[D][D];
      P23 = new double[D][D];
      P123 = new double[D][D][D];

      LogInfo.begin_track( "Moments @" + offset );
      double count = 0.0;
      for( int instance = offset; instance < offset + length; instance++ ) {
        int instanceLength = seqData.getInstanceLength( instance );
        for( int idx = 0; idx < instanceLength - 2; idx++ ) {
          double[] x1 = seqData.getDatum( instance, idx );
          double[] x2 = seqData.getDatum( instance, idx+1 );
          double[] x3 = seqData.getDatum( instance, idx+2 );
          count++;
          // Add into Pairs and Triples
          for( int i = 0; i < D; i++ ) {
            for( int j = 0; j < D; j++ ) {
              P12[i][j] += (x1[i] * x2[j] - P12[i][j])/(count);
              P13[i][j] += (x1[i] * x3[j] - P13[i][j])/(count);
              P23[i][j] += (x2[i] * x3[j] - P23[i][j])/(count);
              for( int k = 0; k < D; k++ ) {
                P123[i][j][k] += (x1[i] * x2[j] * x3[k] - P123[i][j][k])/(count);
              } 
            }
          }
        }
        if( instance % 100 == 0 )
          Execution.putOutput( "Moments @" + offset + " status", ((float)(instance - offset) * 100)/length );
      }
      LogInfo.end_track( "Moments @" + offset );
    }
  }

  /**
   * Compute moments in the embarrasingly parallel way - just split
   * into k parts.
   */
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> 
      run(ProjectedCorpus C) {
    int offset = 0;
    int totalLength = C.C.length;
    int length = totalLength/nThreads;

    Worker[] comps = new Worker[nThreads];
    // Map
    for( int i = 0; i < nThreads; i++ ) {
      comps[i] = new Worker( C, offset, length );
      offset += length;
      comps[i].start();
    }

    // Stall
    for( int i = 0; i < nThreads; i++ ) {
      try {
        comps[i].join();
      } catch (InterruptedException e) {
        LogInfo.logsForce( "Thread was interrupted: ", e.getMessage() );
      }
    }

    // Reduce
    // Average over all the comps
    int D = C.projectionDim;
    double[][] P12 = new double[D][D];
    double[][] P13 = new double[D][D];
    double[][] P23 = new double[D][D];
    double[][][] P123 = new double[D][D][D];
    for(Worker comp : comps) {
      for(int i = 0; i < D; i++) {
        for(int j = 0; j < D; j++) {
          double ratio = (double) comp.length/totalLength;
          P12[i][j] += ratio * comp.P12[i][j];
          P13[i][j] += ratio * comp.P13[i][j];
          P23[i][j] += ratio * comp.P23[i][j];
          for(int k = 0; k < D; k++) {
            P123[i][j][k] += ratio * comp.P123[i][j][k];
          }
        }
      }
    }

    SimpleMatrix P12_ = new SimpleMatrix( P12 );
    SimpleMatrix P13_ = new SimpleMatrix( P13 );
    SimpleMatrix P23_ = new SimpleMatrix( P23 );
    FullTensor P123_ = new FullTensor( P123 );

    return new Quartet<>( P12_, P13_, P23_, P123_ );
  }

  public static class Options {
    @Option(gloss="File containing word-index representation of data", required=true)
      public String dataPath;
    @Option(gloss="File containing word-index to word map", required=true)
      public String mapPath;
    @Option(gloss="Dimensions for random proj.")
      public int randomProjDim = 10;
    @Option(gloss="Seed for random proj.")
      public int randomProjSeed = 1;
  }
  public static Options opts = new Options();

  public void run() {
    // Read corpus
    try {
      ProjectedCorpus PC = new ProjectedCorpus( 
        Corpus.parseText( opts.dataPath, opts.mapPath ),
        opts.randomProjDim, opts.randomProjSeed );
      
      // Compute moments
      Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,FullTensor> moments;
      moments = run(PC);

      // Debug
      Execution.putOutput( "M12", moments.getValue0() );
      Execution.putOutput( "M13", moments.getValue1() );
      Execution.putOutput( "M23", moments.getValue2() );
      Execution.putOutput( "M123", moments.getValue3() );

      // Save
      String outFilename = Execution.getFile( "moments.dat" );
      ObjectOutputStream out = new ObjectOutputStream( new FileOutputStream( outFilename ) ); 
      out.writeObject(opts.randomProjSeed);
      out.writeObject(moments);
      out.close();
    } catch( IOException e ) {
      LogInfo.logs( e );
    }
  }

  public static void main(String[] args) {
    Execution.run(args, new MomentComputer(), "Main", opts);
  }

}
