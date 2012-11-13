/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Vector;

import learning.spectral.MultiViewMixture;
import learning.utils.MatrixFactory;

import org.ejml.simple.SimpleMatrix;

import sun.misc.Sort;

/**
 * Word clustering using a HMM model
 */
public class WordClustering {
	int k;
	int d;
	protected String[] dict;
	protected int[][] C; 
	
	protected WordClustering( Path fname, int k, int d ) {
		this.k = k;
		this.d = d;
		parseCorpus(fname);
	}
	
	protected void parseCorpus( Path fname ) {
		try( BufferedReader reader = Files.newBufferedReader( fname, Charset.defaultCharset() ) ) {
			LinkedList<int[]> C = new LinkedList<>();
			HashMap<String, Integer> map = new HashMap<>();
			int id = 0;
			
		    String line = null;
		    while ((line = reader.readLine()) != null) {
		    	// Chunk up the line 
		    	String[] tokens = line.split(" ");
		    	int[] indices = new int[tokens.length];
		    	
		    	for( int i = 0; i < tokens.length; i++ )
		    	{
		    		if( !map.containsKey( tokens[i] ))
		    			map.put( tokens[i], id++ );
		    		indices[i] = map.get(tokens[i]);
		    	}
		    	C.add( indices );
		    }
		    
		    // Reverse the map
		    this.dict = new String[ map.size() ];
		    for (Map.Entry<String, Integer> entry : map.entrySet()) {
		    	dict[entry.getValue()] = entry.getKey();
		    }
		    this.C = (int[][]) C.toArray();
		} catch (IOException x) {
		    System.err.format("IOException: %s%n", x);
		}
	}
	
	protected SimpleMatrix[] featurize(SimpleMatrix Z) {
		SimpleMatrix X1, X2, X3;
		// Get the total number of words being considered
		int N = 0;
		for( int[] c : C )
			N += c.length - 2;
		
		X1 = MatrixFactory.zeros( N, d );
		X2 = MatrixFactory.zeros( N, d );
		X3 = MatrixFactory.zeros( N, d );
		
		// Populate the entries of Xi
		int offset = 0;
		for( int[] c : C )
		{
			int n = c.length - 2;
			for( int i = 0; i < n; i++ )
			{
				MatrixFactory.setRow( X1, offset + i, MatrixFactory.row(Z, c[i]));
				MatrixFactory.setRow( X2, offset + i, MatrixFactory.row(Z, c[i+1]));
				MatrixFactory.setRow( X3, offset + i, MatrixFactory.row(Z, c[i+2]));
			}
			offset += n;
		}
		
		SimpleMatrix[] X = {X1, X2, X3};
		
		return X;
	}
	
	protected String[][] run() {
		// Map the words onto features
		int N = C.length;
		SimpleMatrix Z = MatrixFactory.randn( N, d );
		SimpleMatrix X[] = featurize( Z );
		
		MultiViewMixture algo = new MultiViewMixture();
		// Get the cluster centers
		SimpleMatrix O_ = algo.sampleRecovery( k, X[0], X[2], X[1] );
		
		// Find the probability of various words appearing by 
		// inverting the transform.
		
		SimpleMatrix O = Z.mult( O_ );
		
		return null;
	}

	/**
	 * Word Clustering using a HMM kind of model
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Get a better argument parsing mechanism
		
		if( args.length != 5 )
		{
			System.out.println("Usage: " + args[0] +" <k> <d> <fname> <ofname>");
			System.exit(1);
		}
		
		int k = Integer.parseInt( args[1] );
		int d = Integer.parseInt( args[2] );
		Path fname = Paths.get(args[3]);
		Path ofname = Paths.get(args[4]);
		
		WordClustering algo = new WordClustering(fname, k, d);
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
