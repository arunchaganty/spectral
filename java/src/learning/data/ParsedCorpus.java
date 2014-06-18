package learning.data;

import fig.basic.Option;
import learning.data.Corpus;

import java.io.*;
import java.util.*;

public class ParsedCorpus extends Corpus {
  public static class Options {
    @Option(gloss="File containing word-index representation of data", required=true)
    public String dataPath;
    @Option(gloss="File containing word-index to word map", required=true)
    public String mapPath;
    @Option(gloss="File containing tag-index of labelled data", required=true)
    public String labelledDataPath;
    @Option(gloss="File containing tag-index to tag map", required=true)
    public String labelledMapPath;
  }

  private static final long serialVersionUID = 3L;
  
  public String[] tagDict;
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
    super(C);
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
   * @return
   * @throws IOException
   */
  public static ParsedCorpus parseText( String seqFilename, String dictFilename, 
      String tagSeqFilename, String tagDictFilename) throws IOException {
    Corpus C = Corpus.parseText( seqFilename, dictFilename ); 

    BufferedReader reader;

    String[] tagDict = readTagDict(tagDictFilename);
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

    int[][] L_ = L.toArray(new int[0][0]);

    return new ParsedCorpus( C, tagDict, L_ );
  }

  public static String[] readTagDict(String tagDictFilename) throws IOException {
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

    return dict.toArray(new String[dict.size()]);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    for(int sentence = 0; sentence < C.length; sentence++) {
      for(int word = 0; word < C[sentence].length; word++) {
        sb.append(dict[C[sentence][word]]).append("_").append(tagDict[L[sentence][word]]);
        sb.append(" ");
      }
      if(sentence < C.length-1)
        sb.append("\n");
    }

    return sb.toString();
  }

  public List<String> toLines() {
    List<String> lines = new ArrayList<>();
    for(int sentence = 0; sentence < C.length; sentence++) {
      StringBuilder sb = new StringBuilder();
      for(int word = 0; word < C[sentence].length; word++) {
        sb.append(dict[C[sentence][word]]).append("_").append(tagDict[L[sentence][word]]);
        sb.append(" ");
      }
      lines.add(sb.toString());
    }

    return lines;
  }

  public String translateTags(int[] tags) {
    StringBuilder sb = new StringBuilder();
    for( int tag : tags )
      sb.append(tagDict[tag]).append(" ");

    return sb.toString().trim();
  }

}
