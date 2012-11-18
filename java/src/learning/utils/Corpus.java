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
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Random;
import java.util.Vector;

import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleMatrix;

/**
 * Stores a corpus in an integer array
 */
public class Corpus {
	public String[] dict;
	public int[][] C; 
	public long projectionSeed;
	public int projectionDim;
	protected Random rnd;
	protected long[] seeds;
	
	public static final String DIGIT_CLASS = "@DIGIT@";
	public static final String LOWER_CLASS = "@LOWER@";
	public static final String UPPER_CLASS = "@UPPER@";
	public static final String MISC_CLASS = "@MISC@";
	
	public Corpus( String[] dict, int[][] C ) {
		this.dict = dict;
		this.C = C;
		this.rnd = new Random(projectionSeed);
		this.seeds = null;
	}
	
	/**
	 * Classify a word into the following categories: DIGIT, LOWER, UPPER, MISC
	 * @param word
	 * @return
	 */
	protected static String classify( String word ) {
		// If the word contains a digit, throw it in the digit bin
		if( word.matches(".*[0-9].*") )
			return DIGIT_CLASS;
		else if( word.matches("^[A-Z][a-z]+") )
			return UPPER_CLASS;
		else if( word.matches("[a-z]+") )
			return LOWER_CLASS;
		else 
			return MISC_CLASS;
	}
	
	protected static String[] pruneDictionary( HashMap<String,Integer> counts, int cutoff ) {
	    Vector<String> keys = new Vector<>();
	    for (Map.Entry<String, Integer> entry : counts.entrySet()) {
	    	if( entry.getValue() > cutoff )
	    		keys.add( entry.getKey() );
	    }
	    Collections.sort( keys );
	    
	    // Add some sentinel words
	    keys.add( DIGIT_CLASS );
	    keys.add( LOWER_CLASS );
	    keys.add( UPPER_CLASS );
	    keys.add( MISC_CLASS );
	    
	    return keys.toArray(new String[0]);
	}
	
	// TODO: Create a lazy read version
	/**
	 * Parse a text file into a corpus.
	 * @param fname
	 * @return
	 * @throws IOException
	 */
	public static Corpus parseText( Path fname, int cutoff ) throws IOException {
		BufferedReader reader = Files.newBufferedReader( fname, Charset.defaultCharset() );
		// In the first pass through the file, collect words and their counts
		HashMap<String, Integer> counts = new HashMap<>();
	    String line = null;
	    while ((line = reader.readLine()) != null) {
	    	// Chunk up the line 
	    	String[] tokens = line.split(" ");
	    	for( String token: tokens ) {
	    		if(!counts.containsKey(token)) 
	    			counts.put( token, 0 );
	    		counts.put( token, counts.get(token) + 1);
	    	}
	    }
	    reader.close();
	    
	    // Prune all words which have too small a count
	    String[] keys = pruneDictionary( counts, cutoff );
	    
	    // Create a map for words 
	    HashMap<String, Integer> map = new HashMap<>();
	    for(int i = 0; i < keys.length; i++)
	    	map.put( keys[i], i);
	    
	    // In the second pass, convert the text into integer indices
	    reader = Files.newBufferedReader( fname, Charset.defaultCharset() );
		LinkedList<int[]> C = new LinkedList<>();
	    while ((line = reader.readLine()) != null) {
	    	// Chunk up the line 
	    	String[] tokens = line.split(" ");
	    	int[] indices = new int[tokens.length];
	    	
	    	for( int i = 0; i < tokens.length; i++ )
	    	{
	    		String token = tokens[i];
	    		// Handle words with very few occurrences
	    		if( !map.containsKey(token) )
	    			token = classify(token);
	    		indices[i] = map.get(token);
	    	}
	    	C.add( indices );
	    }
	    reader.close();
	    
    	int[][] C_ = C.toArray(new int[0][0]);
	    
	    return new Corpus( keys, C_ );
	}
	
	public static Corpus parseText( Path fname ) throws IOException {
		return parseText( fname,  0 );
	}
	
	/**
	 * The projection is in principle entirely determined by this master seed and the dimension d
	 * @param seed
	 */
	public void setProjection(long seed, int d) {
		this.projectionSeed = seed;
		this.projectionDim = d;
	}
	
	/**
	 * Populate the projection table
	 * @param seed
	 */
	protected void cacheProjections() {
		seeds = new long[ dict.length ];
		rnd.setSeed(projectionSeed);
		for(int i = 0; i < dict.length; i++ ) {
			seeds[i] = rnd.nextLong();
		}
	}
	
	/**
	 * Get the feature of this word
	 * @param i
	 * @return
	 */
	public SimpleMatrix getFeatureForWord( int i ) {
		if( seeds == null ) cacheProjections();
		
		rnd.setSeed(seeds[i]);
		DenseMatrix64F x = new DenseMatrix64F(projectionDim, 1);
		for(int j = 0; j < projectionDim; j++ )
			x.set(j, rnd.nextGaussian());
		return SimpleMatrix.wrap(x);
	}
	
	/**
	 * Get the distribution over words for this feature
	 * @param feature
	 * @return
	 */
	public SimpleMatrix getWordDistribution( SimpleMatrix x ) {
		if( seeds == null ) cacheProjections();
		
		DenseMatrix64F z = new DenseMatrix64F(dict.length, 1);
		for(int i = 0; i < dict.length; i++ )
			z.set(i, getFeatureForWord(i).dot( x ) );
		return SimpleMatrix.wrap(z);
	}
	
	
	
}
