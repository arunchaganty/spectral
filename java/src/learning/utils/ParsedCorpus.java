package learning.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

public class ParsedCorpus extends Corpus {
	public String[] Zdict; // The parsed token state dictionary
	public int[][] Z; // The parsed token
	
	public ParsedCorpus( String[] dict, int[][] C, String[] Zdict, int[][] Z ) {
		super( dict, C );
		this.Zdict = Zdict;
		this.Z = Z;
	}
	
	/**
	 * Parse a parsed sentence dictionary. Every line contains a tree represented as S-expression
	 * @param fname
	 * @return
	 * @throws IOException
	 */
	public static ParsedCorpus parseText( Path fname ) throws IOException {
		LinkedList<int[]> C = new LinkedList<>();
		LinkedList<int[]> Z = new LinkedList<>();
		HashMap<String, Integer> wordMap = new HashMap<>();
		HashMap<String, Integer> posMap = new HashMap<>();
		
		BufferedReader reader = Files.newBufferedReader( fname, Charset.defaultCharset() );
	    String line = null;
	    while ((line = reader.readLine()) != null) {
	    	// Chunk up the line 
	    	String[] tokens = line.split(" ");
	    	
	    	LinkedList<Integer> x = new LinkedList<Integer>(); // Words
	    	LinkedList<Integer> z = new LinkedList<Integer>(); // PoS tag
	    	
	    	for( int i = 0; i < tokens.length; i++ ) {
	    		int n = tokens[i].length();
		    	// Because we have S-expressions, we just need to seek for leaf nodes matching "[^)])$"
	    		// HACK: Check for (NP~http://) which just parses weird
	    		if( tokens[i].charAt(0) != '(' && tokens[i].charAt(n-1) == ')' && tokens[i].charAt(n-2) != ')' ) {
	    			String word = tokens[i].substring(0,n-1);
	    			assert( tokens[i-1].charAt(0) == '(' );
	    			String pos = tokens[i-1].substring(1);
	    			
	    			if( !wordMap.containsKey(word) )
	    				wordMap.put( word, wordMap.size() );
	    			x.add( wordMap.get(word) );
	    			if( !posMap.containsKey(pos) )
	    				posMap.put( pos, posMap.size() );
	    			z.add( posMap.get(pos) );
	    		}
	    	}
	    	
	    	if( x.size() > 0 )
	    	{
		    	int[] x_ = Misc.toPrimitive( x.toArray(new Integer[0]) );
		    	int[] z_ = Misc.toPrimitive( z.toArray(new Integer[0]) );
		    	
		    	C.add( x_ ); Z.add( z_ );
	    	}
	    }
	    reader.close();
	    
	    // Reverse the map
	    String[] dict = new String[ wordMap.size() ];
	    for (Map.Entry<String, Integer> entry : wordMap.entrySet()) {
	    	dict[entry.getValue()] = entry.getKey();
	    }
	    String[] Zdict = new String[ posMap.size() ];
	    for (Map.Entry<String, Integer> entry : posMap.entrySet()) {
	    	Zdict[entry.getValue()] = entry.getKey();
	    }
	    
    	int[][] C_ = C.toArray(new int[0][0]);
    	int[][] Z_ = Z.toArray(new int[0][0]);
	    
	    return new ParsedCorpus( dict, C_, Zdict, Z_ );
	}

}
