package learning.data;

import fig.basic.LogInfo;
import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import org.ejml.simple.SimpleMatrix;

/**
 * A number of general purpose routines to compute moments of a large sparse database.
 */
public class MomentComputationWorkers {
  public abstract static class MatrixWorker extends Thread {
    public double[][] M;
    public final Corpus C;

    public final int offset; public final int length;

    public MatrixWorker(Corpus C, int offset, int length) {
      this.C = C;
      this.offset = offset; this.length = length;
    }
  }
  public static interface MatrixWorkerFactory {
    public MatrixWorker createWorker(Corpus C, int offset, int length);
  }

  public abstract static class TensorWorker extends Thread {
    public double[][][] T;
    public final Corpus C;

    public final int offset; public final int length;

    public TensorWorker(Corpus C, int offset, int length) {
      this.C = C;
      this.offset = offset; this.length = length;
    }
  }
  public static interface TensorWorkerFactory {
    public TensorWorker createWorker(Corpus C, int offset, int length);
  }

  public static SimpleMatrix computeMatrix(final Corpus C, final int nThreads, final MatrixWorkerFactory factory) {
    int offset = 0;
    int totalLength = C.C.length;
    int length = totalLength/nThreads;

    MatrixWorker[] comps = new MatrixWorker[nThreads];
    // Map
    for( int i = 0; i < nThreads; i++ ) {
      comps[i] = factory.createWorker(C, offset, Math.min(length, totalLength - offset));
      offset += length;
      comps[i].start();
    }

    // Stall
    for( int i = 0; i < nThreads; i++ ) {
      try {
        comps[i].join();
      } catch (InterruptedException e) {
        LogInfo.logsForce("Thread was interrupted: ", e.getMessage());
      }
    }

    // Reduce
    // Average over all the comps
    int rows = comps[0].M.length;
    int cols = comps[0].M[0].length;
    double[][] P = new double[rows][cols];
    for(MatrixWorker comp : comps) {
      for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
          double ratio = (double) comp.length/totalLength;
          P[i][j] += ratio * comp.M[i][j];
        }
      }
    }

    return new SimpleMatrix(P);
  }
  public static FullTensor computeTensor(final Corpus C, final int nThreads, final TensorWorkerFactory factory) {
    int offset = 0;
    int totalLength = C.C.length;
    int length = totalLength/nThreads;

    TensorWorker[] comps = new TensorWorker[nThreads];
    // Map
    for( int i = 0; i < nThreads; i++ ) {
      comps[i] = factory.createWorker(C, offset, Math.min(length, totalLength - offset));
      offset += length;
      comps[i].start();
    }

    // Stall
    for( int i = 0; i < nThreads; i++ ) {
      try {
        comps[i].join();
      } catch (InterruptedException e) {
        LogInfo.logsForce("Thread was interrupted: ", e.getMessage());
      }
    }

    // Reduce
    // Average over all the comps
    int d1 = comps[0].T.length;
    int d2 = comps[0].T[0].length;
    int d3 = comps[0].T[0][0].length;
    double[][][] T = new double[d1][d2][d3];
    for(TensorWorker comp : comps) {
      for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
          for(int k = 0; k < d3; k++) {
            double ratio = (double) comp.length/totalLength;
            T[i][j][k] += ratio * comp.T[i][j][k];
          }
        }
      }
    }

    return new FullTensor(T);
  }

  public static abstract class CorpusMatrixWorker extends MatrixWorker {
    final int wordOffset1; final int wordOffset2;
    final int rows; final int cols;
    public CorpusMatrixWorker( Corpus C, int rows, int cols, int wordOffset1, int wordOffset2, int offset, int length ) {
      super(C, offset, length);
      this.wordOffset1 = wordOffset1;
      this.wordOffset2 = wordOffset2;
      this.rows = rows;
      this.cols = cols;
    }

    public void run() {
      M = new double[rows][cols];

      LogInfo.begin_track( "Moments @" + offset );
      int count = 0;
      for( int instance = offset; instance < offset + length; instance++ ) {
        int[] c = C.C[instance];
        for( int idx = 0; idx < c.length - 2; idx++ ) {
          int row1 = c[idx + wordOffset1];
          int row2 = c[idx + wordOffset2];
          count++;

          update(row1, row2, count);
        }
        if( instance % 100 == 0 )
          LogInfo.logs( "Moments @" + offset + " status", ((float)(instance - offset) * 100)/length );
      }
      LogInfo.end_track( "Moments @" + offset );
    }

    public abstract void update(int bigram1, int bigram2, int count);
  }

  public static class RightMultiplyMatrixWorker extends CorpusMatrixWorker {
    final SimpleMatrix R;
    public RightMultiplyMatrixWorker( Corpus C, SimpleMatrix R, int wordOffset1, int wordOffset2, int offset, int length ) {
      super(C, R.numRows(), R.numCols(), wordOffset1, wordOffset2, offset, length);
      this.R = R;
    }

    public void update( int bigram1, int bigram2, int count ) {
      // Add e_i omega_j^T to matrix (basically place j-th row of Omega in i-th row).
      for( int col = 0; col < cols; col++ ) {
        M[bigram1][col] += (R.get(bigram2,col) - M[bigram1][col])/count;
      }
    }
  }

  public static class LeftTMultiplyMatrixWorker extends CorpusMatrixWorker {
    final SimpleMatrix Lt;
    public LeftTMultiplyMatrixWorker( Corpus C, SimpleMatrix Lt, int wordOffset1, int wordOffset2, int offset, int length ) {
      super(C, Lt.numCols(), Lt.numRows(), wordOffset1, wordOffset2, offset, length);
      this.Lt = Lt;
    }

    public void update( int bigram1, int bigram2, int count ) {
      // Add omega_i e_j^T to matrix (basically place i-th row of Omega in j-th col).
      for( int row = 0; row < rows; row++ ) {
        M[row][bigram2] += (Lt.get(bigram1,row) - M[row][bigram2])/count;
      }
    }
  }

  public static class DoubleMultiplyMatrixWorker extends CorpusMatrixWorker {
    final SimpleMatrix Lt;
    final SimpleMatrix R;
    public DoubleMultiplyMatrixWorker( Corpus C, SimpleMatrix Lt, SimpleMatrix R, int wordOffset1, int wordOffset2, int offset, int length ) {
      super(C, Lt.numCols(), R.numCols(), wordOffset1, wordOffset2, offset, length);
      this.Lt = Lt;
      this.R = R;
    }

    public void update( int bigram1, int bigram2, int count ) {
      // Add omega_i e_j^T to matrix (basically place i-th row of Omega in j-th col).
      for( int row = 0; row < rows; row++ ) {
        for( int col = 0; row < cols; col++ ) {
          M[row][col] += (Lt.get(bigram1,row) * R.get(bigram2,col)  - M[row][col])/count;
        }
      }
    }
  }

  public static abstract class CorpusTensorWorker extends TensorWorker {
    final int wordOffset1; final int wordOffset2; final int wordOffset3;
    final int d1, d2, d3;
    public CorpusTensorWorker( Corpus C, int d1, int d2, int d3, int wordOffset1, int wordOffset2, int wordOffset3, int offset, int length ) {
      super(C, offset, length);
      this.wordOffset1 = wordOffset1;
      this.wordOffset2 = wordOffset2;
      this.wordOffset3 = wordOffset3;
      this.d1 = d1;
      this.d2 = d2;
      this.d3 = d3;
    }

    public void run() {
      T = new double[d1][d2][d3];

      LogInfo.begin_track( "Moments @" + offset );
      int count = 0;
      for( int instance = offset; instance < offset + length; instance++ ) {
        int[] c = C.C[instance];
        for( int idx = 0; idx < c.length - 2; idx++ ) {
          int row1 = c[idx + wordOffset1];
          int row2 = c[idx + wordOffset2];
          int row3 = c[idx + wordOffset3];
          count++;

          update(row1, row2, row3, count);
        }
        if( instance % 100 == 0 )
          LogInfo.logs( "Moments @" + offset + " status", ((float)(instance - offset) * 100)/length );
      }
      LogInfo.end_track( "Moments @" + offset );
    }

    public abstract void update(int trigram1, int trigram2, int trigram3, int count);
  }

  public static class FullMultiplyTensorWorker extends CorpusTensorWorker {
    final SimpleMatrix M1, M2, M3;
    public FullMultiplyTensorWorker( Corpus C, SimpleMatrix M1, SimpleMatrix M2, SimpleMatrix M3, int wordOffset1, int wordOffset2, int wordOffset3, int offset, int length ) {
      super(C, M1.numCols(), M2.numCols(), M3.numCols(), wordOffset1, wordOffset2, wordOffset3, offset, length);
      this.M1 = M1;
      this.M2 = M2;
      this.M3 = M3;
    }

    public void update( int trigram1, int trigram2, int trigram3, int count ) {
      // Add M_t1 \otimes  M_t2 \otimes  M_t3
      for( int i = 0; i < d1; i++ ) {
        for( int j = 0; j < d2; j++ ) {
          for( int k = 0; k < d3; k++ ) {
            T[i][j][k] += (M1.get(trigram1,i) * M2.get(trigram2,j) * M3.get(trigram3,k)  - T[i][j][k])/count;
          }
        }
      }
    }
  }

  public static MatrixOps.Matrixable matrixable(final Corpus C, final int wordOffset1, final int wordOffset2, final int nThreads) {
    return new MatrixOps.Matrixable() {
      @Override
      public int numRows() {
        return C.getDimension();
      }
      @Override
      public int numCols() {
        return C.getDimension();
      }
      @Override
      public SimpleMatrix rightMultiply(final SimpleMatrix right) {
        return computeMatrix(C, nThreads, new MatrixWorkerFactory() {
          @Override
          public MatrixWorker createWorker(Corpus C, int offset, int length) {
            return new RightMultiplyMatrixWorker(C, right, wordOffset1, wordOffset2, offset, length);
          }
        });
      }
      @Override
      public SimpleMatrix leftMultiply(final SimpleMatrix leftT) {
        return computeMatrix(C, nThreads, new MatrixWorkerFactory() {
          @Override
          public MatrixWorker createWorker(Corpus C, int offset, int length) {
            return new LeftTMultiplyMatrixWorker( C, leftT, wordOffset1, wordOffset2, offset, length );
          }
        });
      }
      @Override
      public SimpleMatrix doubleMultiply(final SimpleMatrix leftT, final SimpleMatrix right) {
        return computeMatrix(C, nThreads, new MatrixWorkerFactory() {
          @Override
          public MatrixWorker createWorker(Corpus C, int offset, int length) {
            return new DoubleMultiplyMatrixWorker( C, leftT, right, wordOffset1, wordOffset2, offset, length );
          }
        });
      }
    };
  }

  public static MatrixOps.Tensorable tensorable(final Corpus C, final int wordOffset1, final int wordOffset2, final int wordOffset3, final int nThreads) {
    return new MatrixOps.Tensorable() {
      @Override
      public int numD1() {
        return C.getDimension();
      }

      @Override
      public int numD2() {
        return C.getDimension();
      }

      @Override
      public int numD3() {
        return C.getDimension();
      }

      @Override
      public FullTensor multiply123(final SimpleMatrix L, final SimpleMatrix M, final SimpleMatrix N) {
        return computeTensor(C, nThreads, new TensorWorkerFactory() {
          @Override
          public TensorWorker createWorker(Corpus C, int offset, int length) {
            return new FullMultiplyTensorWorker(C, L, M, N, wordOffset1, wordOffset2, wordOffset3, offset, length);
          }
        });
      }
    };
  }


}

