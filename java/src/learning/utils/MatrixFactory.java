/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu>
 * 
 */
package learning.utils;

import java.util.Random;

import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.factory.QRDecomposition;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;

/**
 * Common routines to construct a number of different matrices
 */
public class MatrixFactory {
	private static Random rand = new Random();
	
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
	 * Create a matrix of dimension n x m filled with ones
	 * @param n
	 * @param m
	 * @return
	 */
	public static SimpleMatrix ones( int n, int m ) {
		double[][] vals = new double[n][m];
		for( int i = 0; i < n; i++ )
			for( int j = 0; j < n; j++ )
				vals[i][j] = 1.0;
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
	
	/**
	 * Add a vector to each row/col of a matrix
	 * @param n
	 * @param M
	 * @return
	 */
	public static SimpleMatrix vectorPlus( SimpleMatrix M, SimpleMatrix v ) {
		assert( v.numRows() == 1 || v.numCols() == 1 );
		
		int n = M.numRows();
		int d = M.numCols();
		SimpleMatrix N = zeros(n, d);
		
		// Add to all rows of M
		if( v.numRows() == 1 )
		{
			for( int i = 0; i < M.numRows(); i++ )
			{
				for( int j = 0; j < M.numCols(); j++ )
					N.set(i,j, M.get(i,j) + v.get(j));
			}
		}
		else
		{
			for( int j = 0; j < M.numRows(); j++ )
			{
				for( int i = 0; i < M.numCols(); i++ )
					N.set(i,j, M.get(i,j) + v.get(j));
			}
		}
		
		return N;
	}
	
	/**
	 * Stack v n times to form a n x v matrix
	 * @param n
	 * @param M
	 * @return
	 */
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
	
	/**
	 * Compute the empirical second moment
	 * @param X1 - Data with each point as a row
	 * @param X2
	 * @return
	 */
	public static SimpleMatrix Pairs( SimpleMatrix X1, SimpleMatrix X2 ) {
		assert( X1.numRows() == X2.numRows() );
		int N = X1.numRows();
		
		int n = X1.numCols();
		int m = X2.numCols();
		
		SimpleMatrix Y = MatrixFactory.zeros( n, m );
		for( int i = 0; i < N; i++ )
		{
			SimpleMatrix x1 = X1.extractMatrix(i, i+1, 0, n);
			SimpleMatrix x2 = X2.extractMatrix(i, i+1, 0, m).transpose();
			SimpleMatrix Z = x1.kron(x2);
			// Rolling mean
			Y = Y.plus( Z.minus(Y).divide(i+1) );
		}
		
		return Y;
	}
	
	public static SimpleMatrix[] svdk( SimpleMatrix X, int k )
	{
		@SuppressWarnings("unchecked")
		SimpleSVD<SimpleMatrix> UWV = X.svd(false);
		SimpleMatrix U = UWV.getU();
		SimpleMatrix W = UWV.getW();
		SimpleMatrix V = UWV.getV();
		
		// Truncate U, W and V to k-rank
		U = U.extractMatrix(0, SimpleMatrix.END, 0, k);
		W = W.extractMatrix(0, k, 0, k);
		V = V.extractMatrix(0, SimpleMatrix.END, 0, k);
		
		SimpleMatrix[] ret = {U, W, V};
		
		return ret;
	}
	
	/**
	 * Extract i-th row of the matrix X
	 * @param X
	 * @param i
	 * @return
	 */
	public static SimpleMatrix row( SimpleMatrix X, int i ) {
		return X.extractMatrix( i, i+1, 0, SimpleMatrix.END );
	}
	
	/**
	 * Extract i-th column of the matrix X
	 * @param X
	 * @param i
	 * @return
	 */
	public static SimpleMatrix col( SimpleMatrix X, int i ) {
		return X.extractMatrix( 0, SimpleMatrix.END, i, i+1 );
	}
	
	/**
	 * Set the i-th row of the matrix X
	 * @param X
	 * @param i
	 * @return
	 */
	public static void setRow( SimpleMatrix X, int i, SimpleMatrix r ) {
		if( r.numRows() == 1 )
			r = r.transpose();
		assert( X.numCols() == r.numRows() );
		assert( r.numCols() == 1 );
		
		for( int j = 0; j < r.numRows(); j++ )
			X.set( i, j, r.get(j));
	}
	
	/**
	 * Set the rows i to j of the matrix X
	 * @param X
	 * @param i
	 * @param j
	 * @return
	 */
	public static void setRows( SimpleMatrix X, int i, int j, SimpleMatrix r ) {
		assert( r.numRows() == j-i );
		
		for( int x = i; x < j; x++ )
		{
			for( int y = 0; y < r.numCols(); y++ )
				X.set( x, y, r.get(x-i,y));
		}
	}
	/**
	 * Set the i-th column of the matrix X
	 * @param X
	 * @param i
	 * @return
	 */
	public static void setCol( SimpleMatrix X, int i, SimpleMatrix r ) {
		if( r.numRows() == 1 )
			r = r.transpose();
		assert( X.numRows() == r.numRows() );
		assert( r.numCols() == 1 );
		
		for( int j = 0; j < r.numRows(); j++ )
			X.set( j, i, r.get(j));
	}
	
	/**
	 * Generate a random matrix with standard normal entries.
	 * @param d
	 * @return
	 */
	public static SimpleMatrix randn(int m, int n) {
		SimpleMatrix X = zeros(m,n);
		for( int i = 0; i < m; i++)
			for( int j = 0; j < n; j++)
				X.set( i, j, rand.nextGaussian() );
		
		return X;
	}

	/**
	 * Generate a random orthogonal 'd' dimensional matrix, using the
     * the technique described in: Francesco Mezzadri, "How to generate 
     * random matrices from the classical compact groups" 
	 * @param d
	 * @return
	 */
	public static SimpleMatrix randomOrthogonal(int d) {
		SimpleMatrix Z = randn(d,d);
		QRDecomposition<DenseMatrix64F> Z_QR = DecompositionFactory.qr(Z.numRows(), Z.numCols());
		Z_QR.decompose(Z.getMatrix());
		SimpleMatrix Q = SimpleMatrix.wrap( Z_QR.getQ(null, true) );
		SimpleMatrix R = SimpleMatrix.wrap( Z_QR.getR(null, true) ); 
		SimpleMatrix D = diag(R);
		for( int i = 0; i < d; i++)
			D.set(i, D.get(i)/Math.abs(D.get(i)));
		return Q.mult(diag(D));
	}
	
	public static double distance(SimpleMatrix x, SimpleMatrix y)  {
		assert( x.numCols() == y.numCols() );
		assert( x.numRows() == 1 && y.numRows() == 1 );
		
		double d = 0.0;
		for( int i = 0; i < x.numCols(); i++ )
		{
			double d_ = x.get( i ) - y.get(i);
			d += d_ * d_;
		}
		d = Math.sqrt(d);
		
		return d;
	}

	/**
	 * Find the pairwise distance of the i-th row in X and j-th row in Y
	 * @param X
	 * @param Y
	 * @return
	 */
	public static SimpleMatrix cdist(SimpleMatrix X, SimpleMatrix Y) {
		assert( X.numCols() == Y.numCols() );
		
		int n = X.numRows();
		int m = Y.numRows();
		
		SimpleMatrix Z = zeros( n, m );
		
		for( int i = 0; i < n; i++ ) {
			for( int j = 0; j<m; j++ ) {
				// Find the distance between X and Y
				double d = 0;
				for( int k = 0; k < X.numCols(); k++ ) {
					double d_ = X.get(i,k) - X.get(j,k);
					d += d_ * d_;
				}
				Z.set( i, j, Math.sqrt( d ) );
			}
		}
		
		return Z;
	}
}
