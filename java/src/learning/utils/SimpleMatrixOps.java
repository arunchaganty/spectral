/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.utils;

import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;

/**
 * 
 */
public class SimpleMatrixOps {
	
	/**
	 * Project each columns of X onto a simplex
	 * @param X
	 * @return
	 */
	public static int argmax( DenseMatrix64F X )
	{
		int max_i = 0;
		double max = Double.NEGATIVE_INFINITY;
		for( int i = 0; i < X.getNumElements(); i++ ) {
			double x = X.get(i);
			if( x > max ) {
				max = x; max_i = i;
			}
		}
		
		return max_i;
	}	
	public static int argmax( SimpleMatrix X )
	{
		return argmax( X.getMatrix() );
	}
	
	public static SimpleMatrix col( SimpleMatrix X, int i ) {
		return X.extractMatrix( 0, SimpleMatrix.END, i, i+1 );
	}

	public static DenseMatrix64F col( DenseMatrix64F X, int i ) {
		DenseMatrix64F x = new DenseMatrix64F( X.numRows, 1 );
		for( int j = 0; j < X.numRows; j++ )
			x.set(j, X.get( X.getIndex(j, i)));
		return x;
	}
	
	public static SimpleMatrix row( SimpleMatrix X, int i ) {
		return X.extractMatrix( i, i+1, 0, SimpleMatrix.END);
	}

	public static DenseMatrix64F row( DenseMatrix64F X, int i ) {
		DenseMatrix64F x = new DenseMatrix64F( 1, X.numCols );
		for( int j = 0; j < X.numCols; j++ )
			x.set(j, X.get( X.getIndex(i, j)));
		return x;
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
	
	public static void setRow( SimpleMatrix X, int i, SimpleMatrix r ) {
		if( r.numRows() == 1 )
			r = r.transpose();
		assert( X.numCols() == r.numRows() );
		assert( r.numCols() == 1 );

		for( int j = 0; j < r.numRows(); j++ )
			X.set( i, j, r.get(j));
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
		
		SimpleMatrix Z = SimpleMatrixFactory.zeros( n, m );
		
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
	
	public static double columnSum(DenseMatrix64F X, int col ){
		double sum = 0;
		for( int i = 0; i < X.numRows; i++ )
			sum += X.get( X.getIndex( i, col ));
		return sum;
	}
	
	public static double columnSum(SimpleMatrix X, int col ){
		return columnSum( X.getMatrix(), col );
	}
	
	/**
	 * Project each columns of X onto a simplex
	 * @param X
	 * @return
	 */
	public static DenseMatrix64F projectOntoSimplex( DenseMatrix64F X ) {
		int n = X.numRows;
		int m = X.numCols;
		for( int col = 0; col < m; col++ ) {
			double X_bar = columnSum( X, col )/n;
			for( int i = 0; i < n; i++ ) {
				double x  = X.get( X.getIndex(i, col) );
				
				if( X_bar < 0 ) x = -(x + X_bar + 1.0/n);
				else x = x - X_bar + 1.0/n;
				if( x < 0 ) x = 0;
				X.set( X.getIndex(i, col), x);
			}
			// Re-normalize
			double X_sum = columnSum( X, col );
			for( int i = 0; i < n; i++ ) {
				double x  = X.get( X.getIndex(i, col) );
				X.set( X.getIndex(i, col), x/X_sum);
			}
			// Re-normalize
			assert( Math.abs( columnSum( X, col ) - 1.0 ) < 1e-4 );
		}
		
		return X;
	}
	
	/**
	 * Project each columns of X onto a simplex
	 * @param X
	 * @return
	 */
	public static SimpleMatrix projectOntoSimplex( SimpleMatrix X )
	{
		DenseMatrix64F Y = X.getMatrix().copy();
		return new SimpleMatrix( projectOntoSimplex(Y));
	}	
	
    public static SimpleMatrix[] svdk( SimpleMatrix X, int k ) {
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
	
	
	
}
