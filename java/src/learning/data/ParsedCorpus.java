package learning.data;

import learning.data.Corpus;

import java.io.*;
import java.util.*;

public class ParsedCorpus extends Corpus {
  private static final long serialVersionUID = 3L;
  
  String[] tagDict;
  /// Stores an integer representation of labels in the corpus, with each
  /// sentence as a sequence of integer indices into tags.
  public int[][] L; 

  protected ParsedCorpus() {super();}

  public ParsedCorpus(ParsedCorpus C) {
    super((Corpus) C);
    this.tagDict = C.tagDict;
    this.L = C.L;
  }

  public ParsedCorpus( Corpus C, String[] tagDict, int[][] L ) {
    super((Corpus) C);
    this.tagDict = tagDict;
    this.L = L;
  }

  public ParsedCorpus( String[] dict, int[][] C, String[] tagDict, int[][] L ) {
    super( dict, C );
    this.tagDict = tagDict;
    this.L = L;
  }

  public int getTagDimension() {
    return tagDict.length;
  }

  /**
   * Parse a text file into a corpus.
   * @param fname
   * @return
   * @throws IOException
   */
  public static ParsedCorpus parseText( String seqFilename, String dictFilename, 
      String tagSeqFilename, String tagDictFilename) throws IOException {
    Corpus C = Corpus.parseText( seqFilename, dictFilename ); 

    BufferedReader reader;

    // Read file, each line is a word
    LinkedList<String> dict = new LinkedList<String>();
    {
      reader = new BufferedReader( new FileReader( tagDictFilename ) );
      String line = null;
      while ((line = reader.readLine()) != null) {
        // Chunk up the line 
        dict.add( line.trim() );
      }
      reader.close();
    }

    // Read file, each line is a seq of integers (indices into dict)
    LinkedList<int[]> L = new LinkedList<int[]>();
    {
      reader = new BufferedReader( new FileReader( tagSeqFilename ) );
      String line = null;
      while ((line = reader.readLine()) != null) {
        // Chunk up the line 
        String[] tokens = line.split(" ");
        int[] indices = new int[tokens.length];
        for( int i = 0; i < tokens.length; i++ ) {
          indices[i] = Integer.parseInt(tokens[i]); 
        }
        L.add( indices );
      }
      reader.close();
    }

    String[] tagDict_ = dict.toArray(new String[0]);
    int[][] L_ = L.toArray(new int[0][0]);

    return new ParsedCorpus( C, tagDict_, L_ );
  }
}
