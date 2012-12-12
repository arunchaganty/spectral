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

/**
 * Efficiently computes moments of text corpora.
 */
public class MomentComputer {
    final ProjectedCorpus PC;
    double[][] Theta;
    final int nThreads;
    int nClusters;

    public MomentComputer( ProjectedCorpus PC, int nThreads ) {
        this.PC = PC;
        this.nThreads = nThreads;
    }
    public MomentComputer( ProjectedCorpus PC ) {
        this.PC = PC;
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
            if( offset + length > PC.C.length )
                length = PC.C.length - offset;
            this.length = length;
        }

        public void run() {
            int d = PC.projectionDim;
            P12 = new double[d][d];
            P13 = new double[d][d];

            CachedProjectedCorpus CPC = new CachedProjectedCorpus( PC );

            LogInfo.begin_track( "Pairs @" + offset );
            double count = 0.0;
            for( int c_i = offset; c_i < offset + length; c_i++ ) {
                int[] doc = PC.C[c_i];
                int l = doc.length - 2;
                for( int word = 0; word < l; word++ ) {
                    double[] x1 = CPC.featurize( doc[word] );
                    double[] x2 = CPC.featurize( doc[word+1] );
                    double[] x3 = CPC.featurize( doc[word+2] );
                    // Add into P13
                    count++;
                    for( int i = 0; i < d; i++ ) {
                        for( int j = 0; j < d; j++ ) {
                            P12[i][j] += (x1[i] * x2[j] - P12[i][j])/(count);
                            P13[i][j] += (x1[i] * x3[j] - P13[i][j])/(count);
                        }
                    }
                }
                if( c_i % 10 == 0 )
                    Execution.putOutput( "Pairs@" + offset + " status", ((float)c_i * 100)/PC.C.length );
            }
            LogInfo.end_track( "Pairs @" + offset );
        }
    }

    /**
     * Compute moments in the embarrasingly parallel way - just split
     * into k parts.
     */
    public SimpleMatrix[] Pairs() {
        int offset = 0;
        int totalLength = PC.C.length;
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
        int d = PC.projectionDim;
        double[][] P12 = new double[d][d];
        double[][] P13 = new double[d][d];
        for(PairsComputer comp : comps) {
            for(int i = 0; i < d; i++) {
                for(int j = 0; j < d; j++) {
                    double ratio = (double) comp.length/totalLength;
                    P12[i][j] += ratio * comp.P12[i][j];
                    P13[i][j] += ratio * comp.P13[i][j];
                }
            }
        }

        SimpleMatrix P12_ = new SimpleMatrix( P12 );
        SimpleMatrix P13_ = new SimpleMatrix( P13 );
        SimpleMatrix[] P12P13 = {P12_, P13_};

        return P12P13;
    }

    /**
     * Compute moments for a range in the dataset.
     */
    protected class TriplesComputer extends Thread {
        public double[][][] P132;

        public int offset;
        public int length;
        public TriplesComputer( int offset, int length ) {
            this.offset = offset;
            if( offset + length > PC.C.length )
                length = PC.C.length - offset;
            this.length = length;
        }

        public void run() {
            int d = PC.projectionDim;
            P132 = new double[nClusters][d][d];

            CachedProjectedCorpus CPC = new CachedProjectedCorpus( PC );

            LogInfo.begin_track( "Triples @" + offset );
            double count = 0.0;
            for( int c_i = offset; c_i < offset + length; c_i++ ) {
                int[] doc = PC.C[c_i];
                int l = doc.length - 2;
                for( int word = 0; word < l; word++ ) {
                    double[] x1 = CPC.featurize( doc[word] );
                    double[] x2 = CPC.featurize( doc[word+1] );
                    double[] x3 = CPC.featurize( doc[word+2] );

                    // Compute inner products
                    double[] prod = new double[nClusters];
                    for( int i = 0; i < nClusters; i++ )
                        for( int j = 0; j < d; j++ )
                            prod[i] += x2[j] * Theta[i][j];

                    // Add into P132
                    count++;
                    for( int i = 0; i < d; i++ ) {
                        for( int j = 0; j < d; j++ ) {
                            for( int cluster = 0; cluster < nClusters; cluster++ ) {
                                P132[cluster][i][j] += (prod[cluster] * x1[i] * x3[j] - P132[cluster][i][j])/count;
                            }
                        }
                    }
                }
                if( c_i % 10 == 0 )
                    Execution.putOutput( "Triples@" + offset + " status", ((float)c_i * 100)/PC.C.length );
            }
            LogInfo.end_track( "Triples @" + offset );
        }
    }

    /**
     * Compute the projected tensor using each column of theta.
     */
    public SimpleMatrix[] Triples( SimpleMatrix theta ) {
        nClusters = theta.numCols();
        // Let's use rows because they're easier to index with
        Theta = MatrixFactory.toArray( theta.transpose() );

        int offset = 0;
        int totalLength = PC.C.length;
        int length = totalLength/nThreads;

        TriplesComputer[] comps = new TriplesComputer[nThreads];
        for( int i = 0; i < nThreads; i++ ) {
            comps[i] = new TriplesComputer( offset, length );
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
        int d = PC.projectionDim;
        double[][][] P132 = new double[nClusters][d][d];
        for(TriplesComputer comp : comps) {
            for(int cluster = 0; cluster < nClusters; cluster++) {
                for(int i = 0; i < d; i++) {
                    for(int j = 0; j < d; j++) {
                        double ratio = (double) comp.length/totalLength;
                        P132[cluster][i][j] += ratio * comp.P132[cluster][i][j];
                    }
                }
            }
        }
        SimpleMatrix[] P132_ = new SimpleMatrix[nClusters];
        for( int i = 0; i < nClusters; i++ )
            P132_[i] = new SimpleMatrix( P132[i] );

        return P132_;
    }

}
