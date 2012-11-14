/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import learning.spectral.MultiViewMixture;
import learning.utils.Corpus;
import org.ejml.simple.SimpleMatrix;

/**
 * Word clustering using a HMM model
 */
public class WordClustering {
	int k;
	int d;
	protected Corpus C;
	
	protected WordClustering( Corpus C, int k, int d ) {
		this.k = k;
		this.d = d;
		this.C = C;
	}
	
	protected String[][] run() {
		// Map the words onto features
		SimpleMatrix X[] = C.featurize( d );
		
		MultiViewMixture algo = new MultiViewMixture();
		// Get the cluster centers
		SimpleMatrix O_ = algo.sampleRecovery( k, X[0], X[2], X[1] );
		
		// Find the probability of various words appearing by 
		// inverting the transform.
		
		SimpleMatrix O = C.defeaturize( O_ );
		
		return null;
	}

	/**
	 * Word Clustering using a HMM kind of model
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Get a better argument parsing mechanism
		
		if( args.length != 4 )
		{
			System.out.println("Usage: <k> <d> <fname> <ofname>");
			System.exit(1);
		}
		
		int k = Integer.parseInt( args[0] );
		int d = Integer.parseInt( args[1] );
		Path fname = Paths.get(args[2]);
		Path ofname = Paths.get(args[3]);
		
		Corpus C = Corpus.parseText(fname);
		
		WordClustering algo = new WordClustering(C, k, d);
		String[][] clusters = algo.run();
		
		try (BufferedWriter writer = Files.newBufferedWriter(ofname, Charset.defaultCharset())) {
			for( int i = 0; i < clusters.length; i++) {
				writer.write( clusters[i].toString() );
				writer.write("\n");
			}
		} catch (IOException x) {
		    System.err.format("IOException: %s%n", x);
		}
	}

}
