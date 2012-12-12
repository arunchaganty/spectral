/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import learning.exceptions.RecoveryFailure;
import learning.spectral.MultiViewMixture;
import learning.data.Corpus;
import learning.data.ProjectedCorpus;
import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.RandomFactory;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;

import java.util.Date;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.ClassNotFoundException;

/**
 * Word clustering using a HMM model
 */
public class SpectralWordClustering implements Runnable {

	@Option(gloss = "Number of classes", required = true)
	public int k;
	@Option(gloss = "Input filename", required=true)
	public String inputPath;
	@Option(gloss = "Word cutoff")
	public int cutoff = 5;
	@Option(gloss = "If true, compute moments from input file", required = true)
	public boolean computeMoments;
	@Option(gloss = "If true, compute clustering from moments", required = true)
	public boolean computeClustering;
	@Option(gloss = "Number of latent dimensions", required=true)
	public int d;
	@Option(gloss = "Number of attempts to try projecting on different theta", required=true)
	public int attempts = 3;
	@Option(gloss = "Random seed")
	public long seed = (new Date()).getTime();

	protected Corpus C;
	protected ProjectedCorpus PC;

  protected SimpleMatrix P12;
  protected SimpleMatrix P13;
  protected SimpleMatrix[] Theta;
  protected SimpleMatrix[][] P132; // Stores various projects of this.

  protected SimpleMatrix[] Pairs( ProjectedCorpus PC ) {
    int d = PC.projectionDim;
    double[][] P12 = new double[d][d];
    double[][] P13 = new double[d][d];

		LogInfo.begin_track( "Pairs" );
		double count = 0.0;
		for( int c_i = 0; c_i < PC.C.length; c_i++ ) {
			int[] doc = PC.C[c_i];
			int l = doc.length - 2;
			for( int word = 0; word < l; word++ ) {
				double[] x1 = PC.featurize( doc[word] );
				double[] x2 = PC.featurize( doc[word+1] );
				double[] x3 = PC.featurize( doc[word+2] );
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
				Execution.putOutput( "Pairs status", ((float)c_i * 100)/C.C.length );
		}
		LogInfo.end_track( "Pairs" );

    SimpleMatrix P12_ = new SimpleMatrix( P12 );
    SimpleMatrix P13_ = new SimpleMatrix( P13 );
    SimpleMatrix[] P12P13 = {P12_, P13_};

    return P12P13;
  }

  protected SimpleMatrix[] Triples( ProjectedCorpus PC, SimpleMatrix theta ) {
    int nClusters = theta.numRows();
    int d = PC.projectionDim;
    double[][][] P132 = new double[nClusters][d][d];

    double[][] theta_ = MatrixFactory.toArray( theta );

		LogInfo.begin_track( "Triples" );
		double count = 0.0;
		for( int c_i = 0; c_i < PC.C.length; c_i++ ) {
			int[] doc = PC.C[c_i];
			int l = doc.length - 2;
			for( int word = 0; word < l; word++ ) {
				double[] x1 = PC.featurize( doc[word] );
				double[] x2 = PC.featurize( doc[word+1] );
				double[] x3 = PC.featurize( doc[word+2] );

        // Compute inner products
        double[] prod = new double[nClusters];
        for( int i = 0; i < nClusters; i++ )
          for( int j = 0; j < d; j++ )
            prod[i] += x2[j] * theta_[i][j];

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
				Execution.putOutput( "Triples status", ((float)c_i * 100)/C.C.length );
		}
		LogInfo.end_track( "Triples" );

    SimpleMatrix[] P132_ = new SimpleMatrix[nClusters];
    for( int i = 0; i < nClusters; i++ )
      P132_[i] = new SimpleMatrix( P132[i] );

    return P132_;
  }

  /**
   * Compute the moments of the data for attempts # of $theta$
   */
  protected void computeMoments( ProjectedCorpus PC ) {
    // Compute P12, P13
    SimpleMatrix[] P12P13 = Pairs( PC );
    P12 = P12P13[0];
    P13 = P12P13[1];

    SimpleMatrix[] U1WU3 = MatrixOps.svdk( P13, k );
    SimpleMatrix U3 = U1WU3[2];

    for(int i = 0; i < attempts; i++) {
      // Generate a theta and project P132 onto it
      SimpleMatrix theta = RandomFactory.orthogonal( d );
      Theta[i] = theta;
      
      theta = U3.mult( theta );
      // Compute the projected moment 
      P132[i] = Triples( PC, theta );
    }
  }
	
  protected void populateMoments() throws IOException, FileNotFoundException, ClassNotFoundException {
    LogInfo.begin_track("preprocessing");
    if( computeMoments ) {
      // Compute moments by reading the file
			C = Corpus.parseText( inputPath , cutoff);
			LogInfo.logsForce( "parsed corpus" );

      PC = ProjectedCorpus.fromCorpus( C, d );
			LogInfo.logsForce( "projected corpus" );
      
      // Compute the moments
      computeMoments( PC );
    } else {
      // Read the moments from the input file
      ObjectInputStream in = new ObjectInputStream( new FileInputStream( inputPath ) ); 
      PC = (ProjectedCorpus) in.readObject();
      P12 = (SimpleMatrix) in.readObject();
      P13 = (SimpleMatrix) in.readObject();
      P132 = (SimpleMatrix[][]) in.readObject();
      Theta = (SimpleMatrix[]) in.readObject();
      in.close();
    }
    LogInfo.end_track("preprocessing");
  }
	
	@Override
	public void run() {
    // Populate the moment matrices, either by reading stored values or
    // computing from files.
		try {
      populateMoments();
		} catch( IOException e ) {
			LogInfo.error( e.getMessage() );
			return;
		} catch( ClassNotFoundException e ) {
			LogInfo.error( e.getMessage() );
			return;
    }

    if( computeClustering ) {
      // TODO: Use Algorithm A to compute the clusterings
    } else {
      // Just save the moments to disc.
      String outputPath = Execution.getFile( "moments.dat" );
      try{ 
        ObjectOutputStream out = new ObjectOutputStream( new FileOutputStream( outputPath ) ); 
        out.writeObject( PC );
        out.writeObject( P12 );
        out.writeObject( P13 );
        out.writeObject( P132 );
        out.writeObject( Theta );
        out.close();
      } catch (IOException e) {
        LogInfo.error( e.getMessage() );
        return;
      }

    }
		
	}

	/**
	 * Word Clustering using a HMM kind of model
	 * @param args
	 * @throws IOException 
	 * @throws RecoveryFailure 
	 */
	public static void main(String[] args) throws IOException, RecoveryFailure {
		Execution.run( args, new SpectralWordClustering() );
	}
}
