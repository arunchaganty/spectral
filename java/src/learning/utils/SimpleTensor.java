/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.utils;

import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleMatrix;

/**
 * A tensor class that makes it easy to project onto a matrix
 */
public class SimpleTensor {
	protected SimpleMatrix[] X;
	
	/**
	 * Construct a tensor with three views of data
	 * @param X1 - First view; each data point is a row.
	 * @param X2
	 * @param X3
	 */
	public SimpleTensor(SimpleMatrix X1, SimpleMatrix X2, SimpleMatrix X3) {
		// All the Xs should have the same number of rows
		assert( X1.numRows() == X2.numRows() && X2.numRows() == X3.numRows() );
		
		this.X = new SimpleMatrix[3];
		this.X[0] = X1;
		this.X[1] = X2;
		this.X[2] = X3;
	}
	
	/**
	 * Project the tensor onto a matrix by taking an inner product with theta
	 * @param axis
	 * @param theta
	 * @return
	 */
	public SimpleMatrix project( int axis, SimpleMatrix theta )
	{
		assert( 0 <= axis && axis < 3 );
		
		// Select the appropriate index
		int idx1, idx2, idx3 = axis;
		switch( axis ){
			case 0:
				idx1 = 1; idx2 = 2;
				break;
			case 1:
				idx1 = 0; idx2 = 2;
				break;
			case 2:
				idx1 = 0; idx2 = 1;
				break;
			default:
				throw new IndexOutOfBoundsException();
		}
		assert( theta.numRows() == X[idx3].numCols() );
		
		DenseMatrix64F X1_ = X[idx1].getMatrix();
		DenseMatrix64F X2_ = X[idx2].getMatrix();
		DenseMatrix64F X3_ = X[idx3].getMatrix();
		DenseMatrix64F theta_ = theta.getMatrix();
		
		int N = X1_.numRows;
		
		int n = X1_.numCols;
		int m = X2_.numCols;
		int p = X3_.numCols;
		
		DenseMatrix64F Y = new DenseMatrix64F( n, m);
		for( int i = 0; i < N; i++ )
		{
			double prod = 0.0;
			for (int l = 0; l < p; l++) {
				double x3 = X3_.data[ X3_.getIndex(i, l)];
				prod += x3 * theta_.data[l];
			}
			for( int j = 0; j < n; j++ ){
				for (int k = 0; k < m; k++) {
					double x1 = X1_.data[ X1_.getIndex(i, j)];
					double x2 = X2_.data[ X2_.getIndex(i, k)];
					double y = Y.data[Y.getIndex(j, k)];
					
					// Rolling mean
					Y.data[Y.getIndex(j, k)] += (prod*x1*x2 - y)/(i+1);
				}
			}
		}
		
		return new SimpleMatrix(Y);
	}
	

}
