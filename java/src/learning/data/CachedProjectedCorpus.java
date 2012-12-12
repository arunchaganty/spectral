/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.data;

import learning.data.ProjectedCorpus;

import org.ejml.simple.SimpleMatrix;

import com.spaceprogram.kittycache.KittyCache;

/**
 * Caches the features from a projected corpus. Each thread should have
 * it's own.
 */
public class CachedProjectedCorpus extends ProjectedCorpus {
  protected ProjectedCorpus PC;
  protected KittyCache<Integer,double[]> featureCache;
  public CachedProjectedCorpus( ProjectedCorpus PC ) {
      this.PC = PC;
      this.featureCache = new KittyCache<>( 1000 );
  }

  /**
   * Get the d-dimensional feature vectore for the i-th index word
   */
  public double[] featurize( int i ) {
    // Try getting it from the cache
	double[] x = featureCache.get(i);
    // Otherwise call PC to get it
    if( x == null ) {
        x = PC.featurize( i );
        featureCache.put(i, x, -1);
    }
		
    return x;
  }
}

