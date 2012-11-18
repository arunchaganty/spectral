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
import java.nio.file.Paths;
import java.util.Date;
import java.util.Random;
import java.util.Vector;
import java.util.Collections;

import learning.spectral.MultiViewMixture;
import learning.spectral.MultiViewMixture.RecoveryFailure;
import learning.utils.Corpus;
import learning.utils.MatrixFactory;
import learning.utils.Tensor;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import fig.basic.Option;
import fig.basic.OptionsParser;


/**
 * Word clustering using a HMM model
 */
public class WordClustering {
	@Option(gloss = "Number of classes", required=true)
	public int k;
	@Option(gloss = "Number of latent dimensiosn", required=true)
	public int d;
	@Option(gloss = "Word cutoff")
	public int cutoff = 5;
	@Option(gloss = "Input filename", required=true)
	public String inputPath;
	@Option(gloss = "Output clusters", required=true)
	public String outputPath;
	@Option(gloss = "Random seed")
	public long seed = (new Date()).getTime();
	
	protected Corpus C;
	
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
	
	protected SimpleMatrix GetPairs13() {
		int n = C.projectionDim;
		
		SimpleMatrix P13 = new SimpleMatrix( n, n );
		for( int[] c : C.C ) {
			int l = c.length - 2;
			for( int i = 0; i < l; i++ )
			{
				SimpleMatrix x1 = C.getFeatureForWord( c[i] );
				SimpleMatrix x3 = C.getFeatureForWord( c[i+2] );
				// Add into P13
				for( int j = 0; j < n; j++ ) {
					for( int k = 0; k < n; k++ ) {
						double p = P13.get( j, k );
						p += (x1.get(j) * x3.get(k) - p)/(i+1);
						P13.set( j, k, p );
					}
				}
			}
		}
		return P13;
	}
	
	protected SimpleMatrix GetPairs12() {
		int n = C.projectionDim;
		
		SimpleMatrix P12 = new SimpleMatrix( n, n );
		for( int[] c : C.C ) {
			int l = c.length - 2;
			for( int i = 0; i < l; i++ )
			{
				SimpleMatrix x1 = C.getFeatureForWord( c[i] );
				SimpleMatrix x2 = C.getFeatureForWord( c[i+1] );
				// Add into P12
				for( int j = 0; j < n; j++ ) {
					for( int k = 0; k < n; k++ ) {
						double p = P12.get( j, k );
						p += (x1.get(j) * x2.get(k) - p)/(i+1);
						P12.set( j, k, p );
					}
				}
			}
		}
		return P12;
	}
	
	protected class Triples132 implements Tensor {
		Triples132() {
		}
		@Override
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
			
			int n = C.projectionDim;
			SimpleMatrix P132 = new SimpleMatrix( n, n );
			for( int[] c : C.C ) {
				int l = c.length - 2;
				for( int i = 0; i < l; i++ )
				{
					SimpleMatrix x1 = C.getFeatureForWord( c[i+off0] );
					SimpleMatrix x2 = C.getFeatureForWord( c[i+off1] );
					SimpleMatrix x3 = C.getFeatureForWord( c[i+off2] );
					double prod = 0.0;
					for( int j = 0; j < n; j++ ) {
						prod += x3.get(j) * theta.get(j);
					}
							
					// Add into P13
					for( int j = 0; j < n; j++ ) {
						for( int k = 0; k < n; k++ ) {
							double p = P132.get( j, k );
							p += ( prod * x1.get(j) * x2.get(k) - p)/(i+1);
							P132.set( j, k, p );
						}
					}
				}
			}
			return P132;
		}
	
	}
	
	protected SimpleMatrix getTopicsFromMeans( SimpleMatrix O ) {
		SimpleMatrix P = new SimpleMatrix( C.dict.length, O.numCols() );
		for( int i = 0; i < O.numCols(); i++ ) {
			SimpleMatrix o = MatrixFactory.col(O, i);
			MatrixFactory.setCol( P, i, C.getWordDistribution(o) );
		}
		return MatrixFactory.projectOntoSimplex( P ); 
	}
	
	protected String[][] run() throws RecoveryFailure, IOException {
		C = Corpus.parseText(Paths.get( inputPath ), cutoff);
		// Map the words onto features
		C.setProjection(seed, d);
		
		MultiViewMixture algo = new MultiViewMixture();
		
		// Get the moments
		// Get the moments because storing the matrices takes too much memory
		SimpleMatrix P12 = GetPairs12();
		SimpleMatrix P13 = GetPairs13();
		Tensor P132 = new Triples132();
		
		// Get the cluster centers
		SimpleMatrix O_ = algo.recoverM3( k, P13, P12, P132 );
		
		// Find the probability of various words appearing by 
		// inverting the transform.
		SimpleMatrix O = getTopicsFromMeans(O_);
		
		
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
		WordClustering algo = new WordClustering();
		OptionsParser parser = new OptionsParser(algo);
		parser.doParse(args);
		
		String[][] clusters = algo.run();
		
		try (BufferedWriter writer = Files.newBufferedWriter(Paths.get( algo.outputPath ), Charset.defaultCharset())) {
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
