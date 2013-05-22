/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

import learning.data.LazyProjectedCorpus;

import org.ejml.simple.SimpleMatrix;

import com.spaceprogram.kittycache.KittyCache;

/**
 * Caches the features from a projected corpus. Each thread should have
 * it's own.
 */
public class CachedProjectedCorpus extends LazyProjectedCorpus {
  protected ProjectedCorpus PC;
  protected KittyCache<Integer,double[]> featureCache;
  public CachedProjectedCorpus( LazyProjectedCorpus PC, int cacheSize ) {
    super( PC );
    this.featureCache = new KittyCache<>( cacheSize );
  }
  public CachedProjectedCorpus( LazyProjectedCorpus PC ) {
    this( PC, 1000 );
  }

  /**
   * Get the d-dimensional feature vectore for the i-th index word
   */
  public double[] featurize( int i ) {
    // Try getting it from the cache
    double[] x = featureCache.get(i);
    // Otherwise call PC to get it
    if( x == null ) {
      x = super.featurize( i );
      featureCache.put(i, x, -1);
    }

    return x;
  }
}

