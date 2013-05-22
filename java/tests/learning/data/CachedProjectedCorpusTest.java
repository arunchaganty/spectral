/**
 * learning.linalg
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

import learning.linalg.MatrixOps;
import learning.data.Corpus;
import learning.data.ProjectedCorpus;

import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

import java.io.*;
import java.lang.ClassNotFoundException;

import fig.basic.LogInfo;

/**
 * 
 */
public class CachedProjectedCorpusTest {
  static double EPS_ZERO = 1e-7;
  static double EPS_CLOSE = 1e-4;

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
    ProjectedCorpus PC = ProjectedCorpus.fromCorpus( C, 10, 1 );
    CachedProjectedCorpus CPC = new CachedProjectedCorpus( PC, 50000 );

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
      LogInfo.logs("At document#" + i );
    }
  }
}

