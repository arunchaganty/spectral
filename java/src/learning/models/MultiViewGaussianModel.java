/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import java.util.Random;

import learning.utils.MatrixFactory;
import learning.utils.RandomFactory;
import learning.utils.VectorOps;

import org.ejml.simple.SimpleMatrix;

/**
 * A Multi-View Gaussian Model
 */
public class MultiViewGaussianModel {
	
	protected int k;
	protected int d;
	protected int V;
	protected SimpleMatrix w;
	protected SimpleMatrix[] M;
	protected SimpleMatrix[][] S;
	
	protected Random rnd;
	
	/**
	 * A Multi-view Gaussian with means given by the matrix M. The individual covariances are given by S.
	 * @param k
	 * @param d
	 * @param M
	 * @param S
	 */
	public MultiViewGaussianModel( int k, int d, int V, SimpleMatrix w, SimpleMatrix M[], SimpleMatrix[][] S ) {
		assert( S.length == k );
		
		this.k = k;
		this.d = d;
		this.V = V;
		
		for( int i = 0; i < V; i++ )
			assert( M[i].numCols() == k && M[i].numRows() == d );
		
		this.w = w;
		this.M = M;
		this.S = S;
		
		rnd = new Random();
	}
	
	public int getK() {
		return k;
	}

	public int getD() {
		return d;
	}

	public int getV() {
		return V;
	}

	public SimpleMatrix getW() {
		return w;
	}

	public SimpleMatrix[] getM() {
		return M;
	}

	public SimpleMatrix[][] getS() {
		return S;
	}

	public void setSeed( long seed ) {
		rnd.setSeed( seed );
	}
	
	/** Generate N samples **/
	public SimpleMatrix[] sample( int N ) {
		SimpleMatrix[] X = new SimpleMatrix[V];
		
		// Take w * N samples of each class
		for( int i = 0; i < V; i++ )
		{
			int	offset = 0;
			X[i] = MatrixFactory.zeros( N, d );
			
			for( int j = 0; j < k-1; j++ )
			{
				int n = (int) ( w.get(j) * N );
				// TODO: Handle covariance
				// Generate samples from M and with co-variance S
				MatrixFactory.setRows(X[i], offset, offset+n,
						MatrixFactory.vectorPlus( RandomFactory.randn( n, d ), 
								MatrixFactory.col(M[i], j).transpose() ) 
								);
				offset += n;
			}
			int n = N - offset;
			// TODO: Handle covariance
			// Generate samples from M and with co-variance S
			MatrixFactory.setRows(X[i], offset, offset+n,
					MatrixFactory.vectorPlus( RandomFactory.randn( n, d ), 
							MatrixFactory.col(M[i], k-1).transpose() ) 
							);
			offset += n;
		}
		
		return X;
	}
	
	public static enum WeightDistribution {
		Uniform,
		Random
	}
	
	public static enum MeanDistribution {
		Hypercube,
		Random
	}
	
	public static enum CovarianceDistribution {
		Eye,
		Spherical,
		Random
	}
	
	public static MultiViewGaussianModel generate( final int K, final int d, final int V, double sigma, WeightDistribution wDistribution, MeanDistribution MDistribution, CovarianceDistribution SDistribution ) {
		double[][] w = new double[1][K];
		double[][][] M = new double[V][K][d];
		double[][][][] S = new double[V][K][d][d];
		
		switch( wDistribution ) {
			case Uniform:
				for(int i = 0; i < K; i++ ) w[0][i] = 1.0/K;
				break;
			case Random:
				// Generate random values, and then normalise
				for(int i = 0; i < K; i++ ) w[0][i] = Math.abs( RandomFactory.randn(1.0) ); 
				VectorOps.normalize( w[0] );
				break;
		}
		
		switch( MDistribution ) {
			case Hypercube:
				for(int v = 0; v < V; v++ ) {
					// Edges of the hypercube
					for(int i = 0; i < K; i++) {
						for(int j = 0; j < d; j++) {
							if( j == (i + v) % d ) M[v][i][j] = 1.0;
							else M[v][i][j] = -1.0;
						}
					}
				}
			break;
			case Random:
				throw new NoSuchMethodError();
		}
		
		switch( SDistribution ) {
			case Eye:
				for(int v = 0; v < V; v++ ) {
					// Edges of the hypercube
					for(int k = 0; k < K; k++) {
						for(int i = 0; i < d; i++) {
							for(int j = 0; j < d; j++) {
								S[v][k][i][j] = (i == j) ? sigma : 0.0;
							}
						}
					}
				}
			break;
			case Spherical:
				for(int v = 0; v < V; v++ ) {
					// Edges of the hypercube
					for(int k = 0; k < K; k++) {
						for(int i = 0; i < d; i++) {
							for(int j = 0; j < d; j++) {
								S[v][k][i][j] = (i == j) ? Math.abs( RandomFactory.randn(sigma) ) : 0.0;
							}
						}
					}
				}
			break;
			case Random:
				throw new NoSuchMethodError();
		}
		
		SimpleMatrix w_ = new SimpleMatrix(w);
		SimpleMatrix M_[] = new SimpleMatrix[V];
		for(int v = 0; v < V; v++) M_[v] = (new SimpleMatrix( M[v] )).transpose();
		SimpleMatrix S_[][] = new SimpleMatrix[V][K];
		for(int v = 0; v < V; v++) {
			for(int k = 0; k < K; k++) {
				S_[v][k] = new SimpleMatrix( S[v][k] );
			}
		}
		
		return new MultiViewGaussianModel(K, d, V, w_, M_, S_);
	}
	

}
