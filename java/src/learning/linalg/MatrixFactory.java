/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu>
 * 
 */
package learning.linalg;

import org.ejml.simple.SimpleMatrix;

/**
 * Common routines to construct a number of different matrices
 */
public class MatrixFactory {
	
	/**
	 * Create a simple matrix from a uni-dimensional array x
	 * @param x
	 * @return
	 */
	public static SimpleMatrix fromVector( double[] x )
	{
		double[][] x_ = {x};
		return new SimpleMatrix( x_ );
	}
	
	/**
	 * Create a matrix of dimension n x m filled with zeros
	 * @param n
	 * @param m
	 * @return
	 */
	public static SimpleMatrix zeros( int n, int m ) {
		return new SimpleMatrix(n, m);
	}
	
	/**
	 * Create a column vector of dimension n filled with zeros
	 * @param n
	 * @return
	 */
	public static SimpleMatrix zeros( int n ) {
		return new SimpleMatrix(n, 1);
	}
	
	/**
	 * Create a matrix of dimension n x m filled with ones
	 * @param n
	 * @param m
	 * @return
	 */
	public static SimpleMatrix ones( int n, int m ) {
		double[][] vals = new double[n][m];
		for( int i = 0; i < n; i++ )
			for( int j = 0; j < m; j++ )
				vals[i][j] = 1.0;
		return new SimpleMatrix(vals);
	}
	
	/**
	 * Create a column vector of dimension n filled with ones
	 * @param n
	 * @return
	 */
	public static SimpleMatrix ones( int n ) {
		double[][] vals = new double[n][1];
		for( int i = 0; i < n; i++ )
				vals[i][0] = 1.0;
		return new SimpleMatrix(vals);
	}
	
	/**
	 * Create an identity matrix of dimension n x n
	 * @param n
	 * @return
	 */
	public static SimpleMatrix eye( int n ) {
		return SimpleMatrix.identity(n);
	}
	
	/**
	 * If M is a n x n matrix, return the diagonal elements
	 * If M is a n x 1 matrix, return a matrix with the diagonal elements equal to M 
	 * @param M
	 * @return
	 */
	public static SimpleMatrix diag( SimpleMatrix M ) {
		
		// If nx1 or 1xn, construct a 1x1 matrix
		if( M.numRows() == 1 || M.numCols() == 1 )
		{
			// Our standard is column vectors 
			if( M.numRows() == 1 ) M = M.transpose();
			
			int n = M.numRows();
			SimpleMatrix N = eye(n);
			for( int i = 0; i < n; i++ )
				N.set(i,i, M.get(i,0));
			return N;
		}
		else
		{
			// Extract the diagonal elements
			assert( M.numRows() == M.numCols() );
			
			int n = M.numRows();
			SimpleMatrix N =  zeros(n, 1);
			for( int i = 0; i < n; i++ )
				N.set(i,0, M.get(i,i));
			return N;
		}
	}	
	public static SimpleMatrix vectorStack( int n, SimpleMatrix M ) {
		assert( M.numRows() == 1 || M.numCols() == 1 );
		// Our standard is column vectors 
		if( M.numRows() == 1 ) M = M.transpose();

		int d = M.numRows();
		SimpleMatrix N = zeros(n, d);
		for( int i = 0; i < n; i++ )
			for( int j = 0; j < d; j++ )
				N.set(i, j, M.get(j,0));
		return N;
	}
	

}

