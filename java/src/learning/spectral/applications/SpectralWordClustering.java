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
import learning.data.MomentComputer;
import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.RandomFactory;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionsParser;
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
	@Option(gloss = "If true, compute clustering from moments")
	public boolean computeClustering;
	@Option(gloss = "Number of latent dimensions", required=true)
	public int d;
	@Option(gloss = "Number of attempts to try projecting on different theta")
	public int attempts = 3;
	@Option(gloss = "Number of threads")
	public int nThreads = 1;
	@Option(gloss = "Random seed")
	public int seed = (int)(new Date()).getTime();

	protected Corpus C;
	protected ProjectedCorpus PC;

  protected SimpleMatrix P12;
  protected SimpleMatrix P13;
  protected SimpleMatrix[] Theta;
  protected SimpleMatrix[][] P132; // Stores various projects of this.

  /**
   * Compute the moments of the data for attempts # of $theta$
   */
  protected void computeMoments( ProjectedCorpus PC ) {
    MomentComputer comp = new MomentComputer( PC, nThreads );
    // Compute P12, P13
    SimpleMatrix[] P12P13 = comp.Pairs();
    P12 = P12P13[0];
    P13 = P12P13[1];

    SimpleMatrix[] U1WU3 = MatrixOps.svdk( P13, k );
    SimpleMatrix U3 = U1WU3[2];

    Theta = new SimpleMatrix[attempts];
    P132 = new SimpleMatrix[attempts][];

    for(int i = 0; i < attempts; i++) {
      // Generate a theta and project P132 onto it
      SimpleMatrix theta = RandomFactory.orthogonal( k );
      Theta[i] = theta;
      
      theta = U3.mult( theta );
      // Compute the projected moment 
      P132[i] = comp.Triples( theta );
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
			LogInfo.logsForce( "Moments computed" );
    } else {
      // Read the moments from the input file
      ObjectInputStream in = new ObjectInputStream( new FileInputStream( inputPath ) ); 
      PC = (ProjectedCorpus) in.readObject();
      P12 = (SimpleMatrix) in.readObject();
      P13 = (SimpleMatrix) in.readObject();
      P132 = (SimpleMatrix[][]) in.readObject();
      Theta = (SimpleMatrix[]) in.readObject();
      in.close();
      LogInfo.logsForce( "Read moments from file" );
    }
    LogInfo.end_track("preprocessing");
  }

  /**
   * Serialise the computed moments
   */
  public void saveMoments() throws IOException {
    String outputPath = Execution.getFile( "moments.dat" );
    System.out.println( outputPath );

    ObjectOutputStream out = new ObjectOutputStream( new FileOutputStream( outputPath ) ); 
    out.writeObject( PC );
    out.writeObject( P12 );
    out.writeObject( P13 );
    out.writeObject( P132 );
    out.writeObject( Theta );
    out.close();
  }



  /**
   * Use the computed moments to learn an observation matrix of an HMM.
   */
	public void runClustering() {
      // TODO: Use Algorithm A to compute the clusterings
  }
	
	@Override
	public void run() {
    // Set the seed
    RandomFactory.setSeed( seed );
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

    System.out.println( P132[0][0] );
    System.out.println( P132[1][1] );

    if( computeClustering ) {
      runClustering();
    } else {
      // Just save the moments to disc.
      try {
        saveMoments();
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
