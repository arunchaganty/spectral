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
import java.util.*;

import org.ejml.simple.SimpleMatrix;

import org.apache.commons.io.FilenameUtils;

import org.javatuples.*;

/**
 * Computes moments 
 */
public class MomentAggregator extends Thread {
  public final int D;
  public final double[][] P12;
  public final double[][] P13;
  public final double[][] P23;
  public final double[][][] P123;

  Iterator<double[][]> dataSeq;
  public MomentAggregator( int D, Iterator<double[][]> dataSeq ) {
    this.dataSeq = dataSeq;

    this.D = D;
    P12 = new double[D][D];
    P13 = new double[D][D];
    P23 = new double[D][D];
    P123 = new double[D][D][D];
  }

  public void run() {
    LogInfo.begin_track( "Moments" + this.getId() );

    double count = 0.0;
    while( dataSeq.hasNext() ) {
      double[][] datum = dataSeq.next();

      double[] x1 = datum[0];
      double[] x2 = datum[1];
      double[] x3 = datum[2];
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
      if( count % 100 == 0 )
        LogInfo.logs( String.format("Moments (%d): %d", this.getId(), (int)count) );
    }
    LogInfo.end_track( "Moments" + this.getId() );
  }
  public Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,FullTensor> getMoments() {
    SimpleMatrix M12 = new SimpleMatrix(P12);
    SimpleMatrix M13 = new SimpleMatrix(P13);
    SimpleMatrix M23 = new SimpleMatrix(P23);
    FullTensor M123 = new FullTensor(P123);

    return new Quartet<>( M12, M13, M23, M123 );
  }
}

