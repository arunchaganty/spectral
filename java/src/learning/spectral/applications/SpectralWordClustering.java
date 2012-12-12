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
	@Option(gloss = "Number of latent dimensiosn", required=true)
	public int d;
	@Option(gloss = "Random seed")
	public long seed = (new Date()).getTime();

	protected Corpus C;
	protected ProjectedCorpus PC;

  protected SimpleMatrix P12;
  protected SimpleMatrix P13;
  protected SimpleMatrix[] Theta;
  protected SimpleMatrix[][] P132; // Stores various projects of this.


  protected void computeMoments( ProjectedCorpus PC ) {
    // TODO: Compute the moments of the corpus.
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
