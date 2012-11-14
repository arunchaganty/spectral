/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import org.ejml.simple.SimpleMatrix;

/**
 * Stores a corpus in an integer array
 */
public class Corpus {
	public String[] dict;
	public int[][] C; 
	public SimpleMatrix Theta = null;
	
	public Corpus( String[] dict, int[][] C ) {
		this.dict = dict;
		this.C = C;
		this.Theta = null;
	}
	
	public static Corpus parseText( Path fname ) throws IOException {
		LinkedList<int[]> C = new LinkedList<>();
		HashMap<String, Integer> map = new HashMap<>();
		
		BufferedReader reader = Files.newBufferedReader( fname, Charset.defaultCharset() );
	    String line = null;
	    while ((line = reader.readLine()) != null) {
	    	// Chunk up the line 
	    	String[] tokens = line.split(" ");
	    	int[] indices = new int[tokens.length];
	    	
	    	for( int i = 0; i < tokens.length; i++ )
	    	{
	    		if( !map.containsKey( tokens[i] ))
	    			map.put( tokens[i], map.size() );
	    		indices[i] = map.get(tokens[i]);
	    	}
	    	C.add( indices );
	    }
	    reader.close();
	    
	    // Reverse the map
	    String[] dict = new String[ map.size() ];
	    for (Map.Entry<String, Integer> entry : map.entrySet()) {
	    	dict[entry.getValue()] = entry.getKey();
	    }
    	int[][] C_ = (int[][]) C.toArray(new int[0][0]);
	    
	    return new Corpus( dict, C_ );
	}
	
	public SimpleMatrix[] featurize(SimpleMatrix Theta) {
		SimpleMatrix X1, X2, X3;
		
		// Save the featurization for later retrieval
		this.Theta = Theta;
		// Get the total number of words being considered
		int N = 0;
		for( int[] c : C )
			N += c.length - 2;
		int n = Theta.numCols();
		
		X1 = MatrixFactory.zeros( N, n );
		X2 = MatrixFactory.zeros( N, n );
		X3 = MatrixFactory.zeros( N, n );
		
		// Populate the entries of Xi
		int offset = 0;
		for( int[] c : C )
		{
			int l = c.length - 2;
			for( int i = 0; i < l; i++ )
			{
				MatrixFactory.setRow( X1, offset + i, MatrixFactory.row(Theta, c[i]));
				MatrixFactory.setRow( X2, offset + i, MatrixFactory.row(Theta, c[i+1]));
				MatrixFactory.setRow( X3, offset + i, MatrixFactory.row(Theta, c[i+2]));
			}
			offset += l;
		}
		
		SimpleMatrix[] X = {X1, X2, X3};
		
		return X;
	}
	
	public SimpleMatrix defeaturize(SimpleMatrix X) {
		return Theta.mult(X);
	}
	
	/**
	 * Construct a mapping from the dictionary to a random n dimensional subspace
	 * @param n
	 * @return
	 */
	public SimpleMatrix[] featurize(int n) {
		int N = dict.length;
		SimpleMatrix Theta = RandomFactory.randn( N, n );
		return featurize( Theta );
	}
}
