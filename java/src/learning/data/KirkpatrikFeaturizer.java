/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

import java.io.*;
import java.util.*;
import fig.basic.Indexer;
import fig.basic.LogInfo;

/**
 * Features defined in KirkpatrikBouchard2010
 */
public class KirkpatrikFeaturizer implements Featurizer {
  public Indexer<String> featureIndexer = new Indexer<String>();
  Corpus C;

  public KirkpatrikFeaturizer( Corpus C ) {
    this.C = C;
    featureIndexer.add("containsDigit");
    featureIndexer.add("containsHyphen");
    featureIndexer.add("initialCap");
    computeTrigrams(C);
    LogInfo.logs("%d trigrams found", featureIndexer.size() - 3 );
    featureIndexer.lock();
  }

  public int numFeatures() { return featureIndexer.size(); }

  public List<Integer> features(String word) {
    List<Integer> map = new ArrayList<Integer>();
    if( containsDigit( word ) ) map.add( featureIndexer.getIndex("containsDigit") );
    if( containsHyphen( word ) ) map.add( featureIndexer.getIndex("containsHyphen") );
    if( initialCap( word ) ) map.add( featureIndexer.getIndex("initialCap") );
    nGram( word, map );

    return map;
  }

  public void computeTrigrams( Corpus C ) {
    LogInfo.begin_track("computeTrigrams");
    for( String word : C.dict ) {
      for( int i = 0; i < word.length() - 3; i++ )
        featureIndexer.add( word.substring(i, i+3) );
    }
    LogInfo.end_track();
  }

  // 1. Contains Digit
  public boolean containsDigit( String word ) {
    return word.matches( ".*[0-9].*" );
  }
  public boolean containsDigit( int idx ) {
    return containsDigit( C.dict[idx] );
  }
  // 2. Contains Hyphen
  public boolean containsHyphen( String word ) {
    return word.contains("-");
  }
  public boolean containsHyphen( int idx ) {
    return containsHyphen( C.dict[idx] );
  }
  // 3. Initial-Cap
  public boolean initialCap( String word ) {
    return word.matches("[A-Z].*");
  }
  public boolean initialCap( int idx ) {
    return initialCap( C.dict[idx] );
  }
  // 4. N-gram
  public void nGram( String word, List<Integer> map ) {
    for( int i = 0; i < word.length()- 3; i++ ) {
      String trigram = word.substring(i, i+3);
      if( featureIndexer.getIndex( trigram ) != -1 )
        map.add( featureIndexer.getIndex( trigram ) );
    }
  }
}

