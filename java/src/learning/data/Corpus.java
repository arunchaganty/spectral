/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

import java.io.*;
import java.util.*;

/**
 * Stores a corpus in an integer array
 */
public class Corpus {
  private static final long serialVersionUID = 2L;
  /// Stores a list of unique vocabulary
  public String[] dict;

  /// Stores an integer representation of the corpus, with each
  /// sentence as a sequence of integer indices into words.
  public int[][] C; 

  /// Keys for catchall classes
  public static final String DIGIT_CLASS = "@DIGIT@";
  public static final String LOWER_CLASS = "@LOWER@";
  public static final String UPPER_CLASS = "@UPPER@";
  public static final String MISC_CLASS = "@MISC@";
  // TODO: Add support here for auxillary features defined in
  // KirkpatrikBouchard2010

  protected Corpus() {}

  public Corpus(Corpus C) {
    this.dict = C.dict;
    this.C = C.C;
  }

  public Corpus( String[] dict, int[][] C ) {
    this.dict = dict;
    this.C = C;
  }

  public int getInstanceCount() {
    return C.length;
  }
  public int getDimension() {
    return dict.length;
  }

  // TODO: Create a lazy read version
  /**
   * Parse a text file into a corpus.
   * @param fname
   * @return
   * @throws IOException
   */
  public static Corpus parseText( String seqFilename, String dictFilename ) throws IOException {
    BufferedReader reader;

    // Read file, each line is a word
    LinkedList<String> dict = new LinkedList<String>();
    {
      reader = new BufferedReader( new FileReader( dictFilename ) );
      String line = null;
      while ((line = reader.readLine()) != null) {
        // Chunk up the line 
        dict.add( line.trim() );
      }
      reader.close();
    }

    // Read file, each line is a seq of integers (indices into dict)
    LinkedList<int[]> C = new LinkedList<int[]>();
    {
      reader = new BufferedReader( new FileReader( seqFilename ) );
      String line = null;
      while ((line = reader.readLine()) != null) {
        // Chunk up the line 
        String[] tokens = line.split(" ");
        int[] indices = new int[tokens.length];
        for( int i = 0; i < tokens.length; i++ ) {
          indices[i] = Integer.parseInt(tokens[i]); 
        }
        C.add( indices );
      }
      reader.close();
    }

    String[] dict_ = dict.toArray(new String[0]);
    int[][] C_ = C.toArray(new int[0][0]);

    return new Corpus( dict_, C_ );
  }

}
