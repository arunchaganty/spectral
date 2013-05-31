/**
 * learning.linalg
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

import learning.data.Corpus;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

/**
 * 
 */
public class CorpusTest {
  String dataPath;
  String mapPath;

  @Before
  public void setUp() {
    dataPath = "tests/learning/data/test.words";
    mapPath = "tests/learning/data/test.index";
  }

  @Test
  public void parseText() throws IOException {
    Corpus C = Corpus.parseText( dataPath, mapPath );
    Assert.assertTrue( C.C.length == 100 );
  }

  // @Before
  // public void setUp() {
  //   corpusPath = "tests/learning/data/test.txt";
  //     //this.getClass().getResource("/tests/learning/data/test.txt").getPath();
  // }

  // @Test
  // public void parseText() throws IOException {
  //   // Test whether parseText faithfully reproduces the text corpus,
  //   // modulo reduction to sentinel classes
  //   // Small cutoff because the test corpus is so small
  //   Corpus C = Corpus.parseText( corpusPath, 2 );

  //   // Try to reproduce the text
	// 	BufferedReader reader = new BufferedReader( new FileReader( corpusPath ) );
  //   String line = null;
  //   for( int i = 0; i < C.C.length; i++ ) {
  //     line = reader.readLine();
  //     Assert.assertTrue( line != null );

  //     String[] tokens = line.split(" ");
  //     int[] doc = C.C[i];
  //     for( int j = 0; j < doc.length; j++ ) {
  //       int word = doc[j];
  //       // Skip sentinel words
  //       if( C.dict[ word ].startsWith("@") && 
  //             C.dict[ word ].endsWith("@") ) 
  //         continue;

  //       // Check the words match
  //       Assert.assertTrue( C.dict[ word ].trim().equals( tokens[j].trim() ) );
  //     }
  //   }
  //   reader.close();
  // }



}
