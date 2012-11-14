/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import java.util.Random;

import learning.utils.MatrixFactory;
import learning.utils.RandomFactory;

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
	
	

}
