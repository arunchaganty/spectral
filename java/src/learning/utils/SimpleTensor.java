/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.utils;

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
		
		int N = X[idx1].numRows();
		
		int n = X[idx1].numCols();
		int m = X[idx2].numCols();
		
		SimpleMatrix Y = MatrixFactory.zeros( n, m );
		for( int i = 0; i < N; i++ )
		{
			SimpleMatrix x1 = X[idx1].extractMatrix(i, i+1, 0, SimpleMatrix.END);
			SimpleMatrix x2t = X[idx2].extractMatrix(i, i+1, 0, SimpleMatrix.END).transpose();
			SimpleMatrix x3 = X[idx3].extractMatrix(i, i+1, 0, SimpleMatrix.END);
			double k = x3.dot( theta );
			SimpleMatrix Z = x1.kron(x2t).scale(k);
			// Rolling mean
			Y = Y.plus( Z.minus(Y).divide(i+1) );
		}
		
		return Y;
	}

}
