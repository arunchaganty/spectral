/**
 * learning.linalg
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

import learning.linalg.MatrixOps;
import learning.data.Corpus;
import learning.data.ProjectedCorpus;

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
public class ProjectedCorpusTest {
  static double EPS_ZERO = 1e-7;
  static double EPS_CLOSE = 1e-4;

  String corpusPath;

  @Before
  public void setUp() {
    corpusPath = "tests/learning/data/test.txt";
      //this.getClass().getResource("/tests/learning/data/test.txt").getPath();
  }

  @Test
  public void parseText() throws IOException {
    // Test whether parseText faithfully reproduces the text corpus,
    // modulo reduction to sentinel classes
    // Small cutoff because the test corpus is so small
    Corpus C = Corpus.parseText( corpusPath, 2 );
    ProjectedCorpus PC = ProjectedCorpus.fromCorpus( C, 10 );

    // Show that the projected vectors always choose the right word with
    // highest probability
    for( int i = 0; i < C.C.length; i++ ) {
      int[] doc = C.C[i];
      for( int j = 0; j < doc.length; j++ ) {
        int word = doc[j];

        double[] pr = PC.getWordDistribution( PC.featurize( word ) );
        // Check the right word has the maximum probability  
        double prMax = MatrixOps.max( pr );
        Assert.assertTrue( Math.abs( pr[word] - prMax ) < EPS_ZERO );
      }
    }
  }



}
