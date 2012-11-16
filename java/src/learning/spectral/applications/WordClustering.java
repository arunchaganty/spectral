/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.SortedSet;
import java.util.Vector;
import java.util.Collections;

import learning.spectral.MultiViewMixture;
import learning.spectral.MultiViewMixture.RecoveryFailure;
import learning.utils.Corpus;
import learning.utils.MatrixFactory;
import learning.utils.Misc;
import learning.utils.RandomFactory;
import learning.utils.Tensor;

import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import com.sun.xml.internal.bind.v2.schemagen.xmlschema.List;

import sun.misc.Sort;


/**
 * Word clustering using a HMM model
 */
public class WordClustering {
	int k;
	int d;
	protected Corpus C;
	
	protected WordClustering( Corpus C, int k, int d ) {
		this.k = k;
		this.d = d;
		this.C = C;
	}
	
	protected int[][] createClusters( SimpleMatrix O ) {
		// Place a topic into a cluster based on it's strength 
		int k = O.numCols();
		int n = O.numRows();
		
		// Java is god-awful
		Vector<Vector<Pair<Double,Integer>>> clusters = new Vector<Vector<Pair<Double, Integer>>>();
		for( int i  = 0; i < k; i++ ) clusters.add( new Vector<Pair<Double,Integer>>() );
			
		// For each element, push it into the appropriate cluster.
		for( int i  = 0; i < n; i++ ) {
			SimpleMatrix m = MatrixFactory.row( O, i );
			int j = MatrixFactory.argmax( m );
			clusters.get( j ).add( new Pair<Double, Integer>( O.get(i,j), i ) );
		}
		
		// Sort each cluster 
		int[][] clusters_ = new int[k][];
		for( int i = 0; i < k; i++ ) {
			Vector<Pair<Double,Integer>> cluster = clusters.get(i);
			// Sorts in inverse order
			Collections.sort( cluster );
			
			clusters_[i] = new int[cluster.size()];
			// Extract the word index only
			for (int j = 0; j < cluster.size(); j++) 
				clusters_[i][j] = clusters.get(i).get(cluster.size()-1-j).getValue1();
		}
		
		return clusters_;
	}
	
	protected String[][] getWordsForClusters( int[][] clusters_ ) {
		String[][] clusters = new String[k][];
		for(int i = 0; i < k; i++ ) {
			clusters[i] = new String[ clusters_[i].length ];
			for(int j = 0; j < clusters_[i].length; j++ ) {
				clusters[i][j] = C.dict[clusters_[i][j]];
			}
		}
		return clusters;
	}
	
	protected SimpleMatrix GetPairs13(SimpleMatrix Theta) {
		int n = Theta.numCols();
		
		SimpleMatrix P13 = new SimpleMatrix( n, n );
		for( int[] c : C.C ) {
			int l = c.length - 2;
			for( int i = 0; i < l; i++ )
			{
				// Add into P13
				for( int j = 0; j < n; j++ ) {
					for( int k = 0; k < n; k++ ) {
						double p = P13.get( j, k );
						p += ( Theta.get( c[i], j ) * Theta.get( c[i+2], k ) - p)/(i+1);
						P13.set( j, k, p );
					}
				}
			}
		}
		return P13;
	}
	
	protected SimpleMatrix GetPairs12(SimpleMatrix Theta) {
		int n = Theta.numCols();
		
		SimpleMatrix P13 = new SimpleMatrix( n, n );
		for( int[] c : C.C ) {
			int l = c.length - 2;
			for( int i = 0; i < l; i++ )
			{
				// Add into P13
				for( int j = 0; j < n; j++ ) {
					for( int k = 0; k < n; k++ ) {
						double p = P13.get( j, k );
						p += ( Theta.get( c[i], j ) * Theta.get( c[i], k ) - p)/(i+1);
						P13.set( j, k, p );
					}
				}
			}
		}
		return P13;
	}
	
	protected class Triples132 implements Tensor {
		protected SimpleMatrix Theta;
		Triples132( SimpleMatrix Theta ) {
			this.Theta = Theta;
		}
		public SimpleMatrix project( int axis, SimpleMatrix theta ) {
			assert( 0 <= axis && axis < 3 );
			
			// Select the appropriate index
			int off0, off1, off2;
			switch( axis ){
				case 0:
					off0 = 1; off1 = 2; off2 = 0;
					break;
				case 1:
					off0 = 1; off1 = 1; off2 = 2;
					break;
				case 2:
					off0 = 0; off1 = 2; off2 = 1; // We want summing over axis 2 to be over X2 for Triples132
					break;
				default:
					throw new IndexOutOfBoundsException();
			}
			
			int n = Theta.numCols();
			SimpleMatrix P132 = new SimpleMatrix( n, n );
			for( int[] c : C.C ) {
				int l = c.length - 2;
				for( int i = 0; i < l; i++ )
				{
					double prod = 0.0;
					for( int j = 0; j < n; j++ ) {
						prod += Theta.get( c[i + off2], j ) * theta.get(j);
					}
							
					// Add into P13
					for( int j = 0; j < n; j++ ) {
						for( int k = 0; k < n; k++ ) {
							double p = P132.get( j, k );
							p += ( prod * Theta.get( c[i + off0], j ) * Theta.get( c[i + off1], k ) - p)/(i+1);
							P132.set( j, k, p );
						}
					}
				}
			}
			return P132;
		}
	
	}
	
	protected String[][] run() throws RecoveryFailure {
		// Map the words onto features
		C.setProjection(d);
		SimpleMatrix Theta = C.getProjection();
		
		MultiViewMixture algo = new MultiViewMixture();
		
		// Get the moments
		// Get the moments because storing the matrices takes too much memory
		SimpleMatrix P12 = GetPairs12(Theta);
		SimpleMatrix P13 = GetPairs13(Theta);
		Tensor P132 = new Triples132(Theta);
		
		// Get the cluster centers
		SimpleMatrix O_ = algo.recoverM3( k, P13, P12, P132 );
		
		// Find the probability of various words appearing by 
		// inverting the transform.
		
		SimpleMatrix O_unnormalised = C.defeaturize( O_ );
		// Project O onto the "simplex" (\sum_i X_i = 1) and truncate towards zero.
		SimpleMatrix O = MatrixFactory.projectOntoSimplex( O_unnormalised ); 
		
		int[][] clusters_ = createClusters( O );
		
		return getWordsForClusters(clusters_);
	}

	/**
	 * Word Clustering using a HMM kind of model
	 * @param args
	 * @throws IOException 
	 * @throws RecoveryFailure 
	 */
	public static void main(String[] args) throws IOException, RecoveryFailure {
		// TODO Get a better argument parsing mechanism
		
		if( args.length != 4 )
		{
			System.out.println("Usage: <k> <d> <fname> <ofname>");
			System.exit(1);
		}
		
		int k = Integer.parseInt( args[0] );
		int d = Integer.parseInt( args[1] );
		Path fname = Paths.get(args[2]);
		Path ofname = Paths.get(args[3]);
		
		Corpus C = Corpus.parseText(fname);
		
		WordClustering algo = new WordClustering(C, k, d);
		String[][] clusters = algo.run();
		
		try (BufferedWriter writer = Files.newBufferedWriter(ofname, Charset.defaultCharset())) {
			for( int i = 0; i < clusters.length; i++) {
				for( String word : clusters[i] )
					writer.write( word + " " );
				writer.write("\n");
			}
		} catch (IOException x) {
		    System.err.format("IOException: %s%n", x);
		}
	}

}
