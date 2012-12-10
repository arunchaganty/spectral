/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.utils;

import org.ejml.simple.SimpleMatrix;

/**
 * 
 */
public class MatrixFactory {
	
	/**
	 * Create a matrix of dimension n x m filled with zeros
	 * @param n
	 * @param m
	 * @return
	 */
	public static double[][] zeros( int n, int m ) {
		return new double[n][m];
	}
	
	/**
	 * Create a column vector of dimension n filled with zeros
	 * @param n
	 * @return
	 */
	public static double[] zeros( int n ) {
		return new double[n];
	}
	
	/**
	 * Create a matrix of dimension n x m filled with ones
	 * @param n
	 * @param m
	 * @return
	 */
	public static double[][] ones( int n, int m ) {
		double[][] vals = new double[n][m];
		for( int i = 0; i < n; i++ )
			for( int j = 0; j < m; j++ )
				vals[i][j] = 1.0;
		return vals;
	}
	
	/**
	 * Create a column vector of dimension n filled with ones
	 * @param n
	 * @return
	 */
	public static double[] ones( int n ) {
		double[] vals = new double[n];
		for( int i = 0; i < n; i++ )
				vals[i] = 1.0;
		return vals;
	}
	
	/**
	 * Create an identity matrix of dimension n x n
	 * @param n
	 * @return
	 */
	public static double[][] eye( int n ) {
		double[][] vals = new double[n][n];
		for( int i = 0; i < n; i++ ) 
			vals[i][i] = 1.0;
		return vals;
	}
	
	/**
	 * Return the diagonal elements
	 * @param M
	 * @return
	 */
	public static double[] toDiag( double[][] M ) {
		assert( M.length > 0 && M.length == M[0].length );
		int n = M.length;
		double[] vals = new double[n];
		for( int i = 0 ; i < n; i++ )
			vals[i] = M[i][i];
		return vals;
	}	
	
	/**
	 * Return a matrix with the diagonal elements equal to Md
	 * @param M
	 * @return
	 */
	public static double[][] toDiag( double[] Md ) {
		int n = Md.length;
		double[][] M = new double[n][n];
		for( int i = 0 ; i < n; i++ )
			M[i][i] = Md[i];
		return M;
	}	
	
	public static double[][] fromSimpleMatrix( SimpleMatrix X ) {
		int r = X.numRows();
		int c = X.numCols();
		double[] data = X.getMatrix().data;
		double[][] result = new double[r][c];
		for(int i = 0; i < r; i++ ) {
			for(int j = 0; j < c; j++ ) {
				result[i][j] = data[i*c + j];
			}
		}
		
		return result;
	}
	
}
