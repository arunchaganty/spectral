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
import java.util.Vector;
import java.util.Collections;

import learning.spectral.MultiViewMixture;
import learning.spectral.MultiViewMixture.RecoveryFailure;
import learning.utils.Corpus;
import learning.utils.SimpleMatrixFactory;
import learning.utils.SimpleMatrixOps;
import learning.utils.Tensor;
import learning.utils.MatrixOps;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;


/**
 * Word clustering using a HMM model
 */
public class WordClustering implements Runnable {
	@Option(gloss = "Number of classes", required=true)
	public int k;
	@Option(gloss = "Number of latent dimensiosn", required=true)
	public int d;
	@Option(gloss = "Word cutoff")
	public int cutoff = 5;
	@Option(gloss = "Input filename", required=true)
	public String inputPath;
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
			SimpleMatrix m = SimpleMatrixOps.row( O, i );
			int j = SimpleMatrixOps.argmax( m );
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
		LogInfo.begin_track( "P13" );
		
		double[][] P13 = new double[n][n];
		double count = 0.0;
		for( int c_i = 0; c_i < C.C.length; c_i++ ) {
			int[] c = C.C[c_i];
			int l = c.length - 2;
			for( int i = 0; i < l; i++ )
			{
				double[] x1 = C.getFeatureForWord( c[i] );
				double[] x3 = C.getFeatureForWord( c[i+2] );
				// Add into P13
				for( int j = 0; j < n; j++ ) {
					for( int k = 0; k < n; k++ ) {
						P13[j][k] += (x1[j] * x3[k] - P13[j][k])/(++count);
					}
				}
			}
			Execution.putOutput( "P13 status", c_i + "/" + C.C.length );
		}
		LogInfo.end_track( "P13" );
		return new SimpleMatrix(P13);
	}
	
	protected void GetPairs(double[][] P12, double [][] P13) {
		int d = C.projectionDim;
		
		LogInfo.begin_track( "Pairs" );
		double count = 0.0;
		for( int c_i = 0; c_i < C.C.length; c_i++ ) {
			int[] c = C.C[c_i];
			int l = c.length - 2;
			for( int i = 0; i < l; i++ )
			{
				double[] x1 = C.getFeatureForWord( c[i] );
				double[] x2 = C.getFeatureForWord( c[i+1] );
				double[] x3 = C.getFeatureForWord( c[i+2] );
				// Add into P13
				count++;
				for( int j = 0; j < d; j++ ) {
					for( int k = 0; k < d; k++ ) {
						P12[j][k] += (x1[j] * x2[k] - P12[j][k])/(count);
						P13[j][k] += (x1[j] * x3[k] - P13[j][k])/(count);
					}
				}
			}
			if( c_i % 10 == 0 )
				Execution.putOutput( "Pairs status", ((float)c_i * 100)/C.C.length );
		}
		LogInfo.end_track( "Pairs" );
	}
	
	protected class Triples132 implements Tensor {
		Triples132() {
		}
		@Override
		public SimpleMatrix project( int axis, SimpleMatrix theta ) {
			assert( 0 <= axis && axis < 3 );
			double[] theta_ = theta.getMatrix().data;
			
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
			
			LogInfo.begin_track( "P132" );
			int d = C.projectionDim;
			double[][] P132 = new double[d][d];
			double count = 0.0;
			for( int c_i = 0; c_i < C.C.length; c_i++ ) {
				int[] c = C.C[c_i];
				int l = c.length - 2;
				for( int i = 0; i < l; i++ )
				{
					double[] x1 = C.getFeatureForWord( c[i+off0] );
					double[] x2 = C.getFeatureForWord( c[i+off1] );
					double[] x3 = C.getFeatureForWord( c[i+off2] );
					double prod = MatrixOps.dot( x2, theta_);
							
					count++;
					// Add into P13
					for( int j = 0; j < d; j++ ) {
						for( int k = 0; k < d; k++ ) {
							P132[j][k] += (prod * x1[j] * x3[k] - P132[j][k])/count;
						}
					}
				}
				Execution.putOutput( "P132 status", ((float)c_i * 100)/C.C.length );
			}
			LogInfo.end_track( "P132" );
			return new SimpleMatrix(P132);
		}
	
	}
	
	protected SimpleMatrix getTopicsFromMeans( SimpleMatrix O ) {
		SimpleMatrix P = new SimpleMatrix( C.dict.length, O.numCols() );
		for( int i = 0; i < O.numCols(); i++ ) {
			SimpleMatrix o = SimpleMatrixOps.col(O, i);
			SimpleMatrixOps.setCol( P, i, C.getWordDistribution(o) );
		}
		return SimpleMatrixOps.projectOntoSimplex( P ); 
	}
	
	@Override
	public void run() {
		try {
			LogInfo.begin_track("preprocessing");
			C = Corpus.parseText(Paths.get( inputPath ), cutoff);
			LogInfo.logsForce( "Parsed corpus" );
			// Map the words onto features
			C.setProjection(seed, d);
			LogInfo.end_track("preprocessing");
		} catch( IOException e ) {
			LogInfo.error( e.getMessage() );
			return;
		}
		
		MultiViewMixture algo = new MultiViewMixture();
		// Get the moments
		// Get the moments because storing the matrices takes too much memory
		double[][] P12_ = new double[d][d];
		double[][] P13_ = new double[d][d];
		GetPairs(P12_, P13_);
		SimpleMatrix P12 = new SimpleMatrix( P12_ );
		SimpleMatrix P13 = new SimpleMatrix( P13_ );
		Tensor P132 = new Triples132();
		SimpleMatrix O;
		
		// Get the cluster centers
		try {
			SimpleMatrix O_ = algo.recoverM3( k, P13, P12, P132 );
			// Find the probability of various words appearing by 
			// inverting the transform.
			O = getTopicsFromMeans(O_);
		} catch( RecoveryFailure e ) {
			LogInfo.errors( "Could not recover M3:", e.getMessage() );
			return;
		}
		
		LogInfo.begin_track("postprocessing");
		int[][] clusters_ = createClusters( O );
		String[][] wordClusters = getWordsForClusters(clusters_);
		LogInfo.end_track("postprocessing");
		
		String outputPath = Execution.getFile("output.clusters");
		
		try (BufferedWriter writer = Files.newBufferedWriter(Paths.get( outputPath ), Charset.defaultCharset())) {
			for( int i = 0; i < wordClusters.length; i++) {
				for( String word : wordClusters[i] )
					writer.write( word + " " );
				writer.write("\n");
			}
		} catch (IOException x) {
		    System.err.format("IOException: %s%n", x);
		}
	}

	/**
	 * Word Clustering using a HMM kind of model
	 * @param args
	 * @throws IOException 
	 * @throws RecoveryFailure 
	 */
	public static void main(String[] args) throws IOException, RecoveryFailure {
		Execution.run( args, new WordClustering() );
	}
}
