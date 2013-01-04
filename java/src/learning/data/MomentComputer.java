/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

import learning.data.Corpus;
import learning.data.ProjectedCorpus;
import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.RandomFactory;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.exec.Execution;

import org.ejml.simple.SimpleMatrix;

import org.javatuples.*;

/**
 * Efficiently computes moments of text corpora.
 */
public class MomentComputer {
  final int nThreads;
  final protected RealSequenceData seqData;

  public MomentComputer( RealSequenceData seqData, int nThreads ) {
    this.seqData = seqData;
    this.nThreads = nThreads;
  }
  public MomentComputer( RealSequenceData seqData ) {
    this.seqData = seqData;
    this.nThreads = 4;
  }

  /**
   * Compute moments for a range in the dataset.
   */
  protected class PairsComputer extends Thread {
    public double[][] P12;
    public double[][] P13;

    public int offset;
    public int length;
    public PairsComputer( int offset, int length ) {
      this.offset = offset;
      if( offset + length > seqData.getInstanceCount() )
        length = seqData.getInstanceCount() - offset;
      this.length = length;
    }

    public void run() {
      int D = seqData.getDimension();
      P12 = new double[D][D];
      P13 = new double[D][D];

      LogInfo.begin_track( "Pairs @" + offset );
      double count = 0.0;
      for( int instance = offset; instance < offset + length; instance++ ) {
        int instanceLength = seqData.getInstanceLength( instance );
        for( int idx = 0; idx < instanceLength - 2; idx++ ) {
          double[] x1 = seqData.getDatum( instance, idx );
          double[] x2 = seqData.getDatum( instance, idx+1 );
          double[] x3 = seqData.getDatum( instance, idx+2 );
          // Add into P13
          count++;
          for( int i = 0; i < D; i++ ) {
            for( int j = 0; j < D; j++ ) {
              P12[i][j] += (x1[i] * x2[j] - P12[i][j])/(count);
              P13[i][j] += (x1[i] * x3[j] - P13[i][j])/(count);
            }
          }
        }
        if( instance % 10 == 0 )
          Execution.putOutput( "Pairs@" + offset + " status", ((float)(instance - offset) * 100)/length );
      }
      LogInfo.end_track( "Pairs @" + offset );
    }
  }

  /**
   * Compute moments in the embarrasingly parallel way - just split
   * into k parts.
   */
  public Pair<SimpleMatrix, SimpleMatrix> Pairs() {
    int offset = 0;
    int totalLength = seqData.getInstanceCount();
    int length = totalLength/nThreads;

    PairsComputer[] comps = new PairsComputer[nThreads];
    for( int i = 0; i < nThreads; i++ ) {
      comps[i] = new PairsComputer( offset, length );
      offset += length;
      comps[i].start();
    }

    for( int i = 0; i < nThreads; i++ ) {
      try {
        comps[i].join();
      } catch (InterruptedException e) {
        LogInfo.logsForce( "Thread was interrupted: ", e.getMessage() );
      }
    }

    // Average over all the comps
    int D = seqData.getDimension();
    double[][] P12 = new double[D][D];
    double[][] P13 = new double[D][D];
    for(PairsComputer comp : comps) {
      for(int i = 0; i < D; i++) {
        for(int j = 0; j < D; j++) {
          double ratio = (double) comp.length/totalLength;
          P12[i][j] += ratio * comp.P12[i][j];
          P13[i][j] += ratio * comp.P13[i][j];
        }
      }
    }

    SimpleMatrix P12_ = new SimpleMatrix( P12 );
    SimpleMatrix P13_ = new SimpleMatrix( P13 );

    return new Pair<>( P12_, P13_ );
  }

  /**
   * Compute moments for a range in the dataset.
   */
  protected class TriplesComputer extends Thread {
    public double[][][] P123;
    public double[][] Theta;
    public int idx1, idx2, idx3;

    public int offset;
    public int length;
    public TriplesComputer( int offset, int length, int axis, double[][] Theta ) {
      assert( 0 <= axis && axis < 3 );

      // Align the axes
      idx1 = idx2 = idx3 = axis;
      switch( axis ){
        case 0: idx1 = 1; idx2 = 2; break;
        case 1: idx1 = 0; idx2 = 2; break;
        case 2: idx1 = 0; idx2 = 1; break;
        default:
                throw new IndexOutOfBoundsException();
      }

      this.offset = offset;
      if( offset + length > seqData.getInstanceCount() )
        length = seqData.getInstanceCount() - offset;
      this.length = length;
      this.Theta = Theta;
    }

    public void run() {
      int D = seqData.getDimension();
      int K = Theta.length;
      P123 = new double[K][D][D];

      LogInfo.begin_track( "Triples @" + offset );
      double count = 0.0;
      for( int instance = offset; instance < offset + length; instance++ ) {
        int instanceLength = seqData.getInstanceLength( instance );
        for( int idx = 0; idx < instanceLength - 2; idx++ ) {
          double[] x1 = seqData.getDatum( instance, idx + idx1 );
          double[] x2 = seqData.getDatum( instance, idx + idx2 );
          double[] x3 = seqData.getDatum( instance, idx + idx3 );

          // Compute inner products
          double[] prod = new double[K];
          for( int i = 0; i < K; i++ )
            for( int j = 0; j < D; j++ )
              prod[i] += x3[j] * Theta[i][j];

          // Add into P123
          count++;
          for( int i = 0; i < D; i++ ) {
            for( int j = 0; j < D; j++ ) {
              for( int cluster = 0; cluster < K; cluster++ ) {
                P123[cluster][i][j] += (prod[cluster] * x1[i] * x2[j] - P123[cluster][i][j])/count;
              }
            }
          }
        }
        if( instance % 10 == 0 )
          Execution.putOutput( "Triples@" + offset + " status", ((float)(instance - offset) * 100)/length );
      }
      LogInfo.end_track( "Triples @" + offset );
    }
  }

  /**
   * Compute the projected tensor using each column of theta.
   *
   * TODO: Support ordering
   */
  public SimpleMatrix[] Triples( int axis, SimpleMatrix theta ) {
    int K = theta.numCols();
    // Let's use rows because they're easier to index with
    double[][] Theta = MatrixFactory.toArray( theta.transpose() );

    int offset = 0;
    int totalLength = seqData.getInstanceCount();
    int length = totalLength/nThreads;

    TriplesComputer[] comps = new TriplesComputer[nThreads];
    for( int i = 0; i < nThreads; i++ ) {
      comps[i] = new TriplesComputer( offset, length, axis, Theta );
      offset += length;
      comps[i].start();
    }

    for( int i = 0; i < nThreads; i++ ) {
      try {
        comps[i].join();
      } catch (InterruptedException e) {
        LogInfo.logsForce( "Thread was interrupted: ", e.getMessage() );
      }
    }

    // Average over all the comps
    int D = seqData.getDimension();
    double[][][] P123 = new double[K][D][D];
    for(TriplesComputer comp : comps) {
      for(int cluster = 0; cluster < K; cluster++) {
        for(int i = 0; i < D; i++) {
          for(int j = 0; j < D; j++) {
            double ratio = (double) comp.length/totalLength;
            P123[cluster][i][j] += ratio * comp.P123[cluster][i][j];
          }
        }
      }
    }
    SimpleMatrix[] P123_ = new SimpleMatrix[K];
    for( int i = 0; i < K; i++ ) {
      P123_[i] = new SimpleMatrix( P123[i] );
    }

    return P123_;
  }

}
