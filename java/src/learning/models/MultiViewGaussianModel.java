/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import java.util.Random;

import learning.utils.SimpleMatrixFactory;
import learning.utils.RandomFactory;
import learning.utils.MatrixOps;
import learning.utils.SimpleMatrixOps;

import fig.prob.MultGaussian;

/**
 * A Multi-View Gaussian Model
 */
public class MultiViewGaussianModel {
	
	protected int K;
	protected int D;
	protected int V;
	protected double[] w;
	protected double[][][] M;
	protected double[][][][] S;
	
	Random rnd = new Random();
	
	/**
	 * A Multi-view Gaussian with means given by the matrix M. The individual covariances are given by S.
	 * @param k
	 * @param d
	 * @param M
	 * @param S
	 */
	public MultiViewGaussianModel( int K, int D, int V, double[] w, double M[][][], double[][][][] S ) {
		assert( S.length == V ); assert( S[0].length == K );
		for( int i = 0; i < V; i++ )
			assert( M[i].length == K && M[i][0].length == D );
		
		this.K = K;
		this.D = D;
		this.V = V;
		
		this.w = w;
		this.M = M;
		this.S = S;
	}
	
	public int getK() {
		return K;
	}

	public int getD() {
		return D;
	}

	public int getV() {
		return V;
	}

	public double[] getW() {
		return w;
	}

	public double[][][] getM() {
		return M;
	}

	public double[][][][] getS() {
		return S;
	}
	
	public void setSeed( long seed ) {
		rnd.setSeed( seed );
	}

	/**
	 * Sample from a particular cluster
	 * @param N
	 * @param cluster
	 * @return
	 */
	public double[][] sample( int n, int view, int cluster ) {
		fig.prob.MultGaussian x = new MultGaussian(M[view][cluster], S[view][cluster]);
		double[][] y = new double[n][D];
		for(int i = 0; i < n; i++)
			y[i] = x.sample(rnd);
		return y;
	}
	
	/** Generate N samples **/
	public double[][][] sample( int N ) {
		double[][][] X = new double[V][N][D];
		
		// Take w * N samples of each class
		for( int v = 0; v < V; v++ )
		{
			int	offset = 0;
			for( int k = 0; k < K-1; k++ )
			{
				int n = (int) ( w[k] * N );
				MatrixOps.setRows(X[v], offset, offset+n, sample(n, v, k) );
				offset += n;
			}
			int n = N - offset;
			MatrixOps.setRows(X[v], offset, offset+n, sample(n, v, K-1) );
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
		double[] w = new double[K];
		double[][][] M = new double[V][K][d];
		double[][][][] S = new double[V][K][d][d];
		
		switch( wDistribution ) {
			case Uniform:
				for(int i = 0; i < K; i++ ) w[i] = 1.0/K;
				break;
			case Random:
				// Generate random values, and then normalise
				for(int i = 0; i < K; i++ ) w[i] = Math.abs( RandomFactory.randn(1.0) ); 
				MatrixOps.normalize( w );
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
		
		return new MultiViewGaussianModel(K, d, V, w, M, S);
	}
	

}
