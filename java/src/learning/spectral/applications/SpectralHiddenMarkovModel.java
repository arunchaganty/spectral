/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import learning.exceptions.RecoveryFailure;
import learning.exceptions.NumericalException;

import learning.models.RealHiddenMarkovModel;

import learning.spectral.MultiViewMixture;
import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.RandomFactory;
import learning.linalg.Tensor;

import learning.data.MomentComputer;
import learning.data.RealSequence;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.*;

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
public class SpectralHiddenMarkovModel implements Runnable {
	@Option(gloss = "Number of classes", required = true)
	public int K;
	@Option(gloss = "Number of latent dimensions", required=true)
	public int D;
	@Option(gloss = "Input filename", required=true)
	public String inputPath;
	@Option(gloss = "Number of Threads to use")
	public int nThreads = 1;
	@Option(gloss = "Random seed")
	public int seed = (int)(new Date()).getTime();

  public SpectralHiddenMarkovModel() {}

  public SpectralHiddenMarkovModel( int K, int D ) {
    this.K = K;
    this.D = D;
  }

  public static Triplet<SimpleMatrix, SimpleMatrix, Tensor> computeExactMoments( RealHiddenMarkovModel model ) {
    SimpleMatrix pi = new SimpleMatrix( model.getPi() );
    SimpleMatrix T = new SimpleMatrix( model.getT() );
    SimpleMatrix O = model.getRealO();

    SimpleMatrix weights = T.mult( pi );
    SimpleMatrix M1 = O.mult( MatrixFactory.diag( pi ) ).mult( T.transpose()
        ).mult( MatrixFactory.diag( weights ).invert() );
    SimpleMatrix M2 = O;
    SimpleMatrix M3 = O.mult( T );

    return MultiViewMixture.computeExactMoments( weights, M1, M2, M3 );
  }

  /**
   * Compute the moments by segmenting the sequence data into Triples
   */
  public Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix[]> partitionData( double[][][] sequenceData, SimpleMatrix theta ) {
    RealSequence seq = new RealSequence( sequenceData );
    MomentComputer comp = new MomentComputer( seq, nThreads );
    // Compute P12, P13
    Pair<SimpleMatrix, SimpleMatrix> P12P13 = comp.Pairs();
    SimpleMatrix P12 = P12P13.getValue0();
    SimpleMatrix P13 = P12P13.getValue1();

    SimpleMatrix U2 = MatrixOps.svdk( P12, K )[2];
   
    SimpleMatrix Theta = U2.mult( theta );
    // Compute the projected moment 
    SimpleMatrix[] P132 = comp.Triples( 1, Theta );

    return new Triplet<>( P12, P13, P132 );
  }

  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix[], SimpleMatrix> partitionData( double[][][] sequenceData ) {
    // Generate a theta and project P312 onto it
    SimpleMatrix theta = RandomFactory.orthogonal( K );
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix[]> result = partitionData( sequenceData, theta );

    return new Quartet<>( result.getValue0(), result.getValue1(), result.getValue2(), theta );
  }

  public SimpleMatrix run(double[][][] data) throws NumericalException {
    // Set the seed
    RandomFactory.setSeed( seed );
    // Compute the moments
    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix[], SimpleMatrix> moments = partitionData( data );

    SimpleMatrix P12 = moments.getValue0();
    SimpleMatrix P13 = moments.getValue1();
    SimpleMatrix[] P132T = moments.getValue2();
    SimpleMatrix Theta = moments.getValue3();

		MultiViewMixture algo = new MultiViewMixture();
    SimpleMatrix O = algo.algorithmB( K, P13, P12, P132T, Theta );
    return O; 
  }
	
	@Override
	public void run() {
    // Read data from a file
    double[][][] data;

    try {
      ObjectInputStream in = new ObjectInputStream( new FileInputStream( inputPath ) ); 
      data = (double[][][]) in.readObject();
      in.close();
    } catch( ClassNotFoundException | IOException e ) {
      System.err.println( e.getMessage() );
      return;
    }

    try{ 
      SimpleMatrix O = run( data );
      System.out.println( O );
    } catch( NumericalException e ) {
      System.err.println( e.getMessage() );
      return;
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

